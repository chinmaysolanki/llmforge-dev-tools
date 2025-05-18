import os
import ast
import json
import shutil
import tempfile
import zipfile
import io # Added for in-memory ZIP file
from pathlib import Path
from typing import List, Optional, Dict, Any, Literal

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse # Added StreamingResponse
from pydantic import BaseModel
from radon.complexity import cc_visit
from radon.metrics import mi_visit
from radon.raw import analyze as raw_analyze
import openai # Will be configured for OpenRouter
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file

app = FastAPI(title="LLMForge – where AI models forge your dev tools.")

# Mount static files
# app.mount("/static", StaticFiles(directory="static", follow_symlink=True), name="static") 
# Vercel will handle static files based on vercel.json and file structure.
# If you have a /static directory in your root, ensure vercel.json routes it.

# In-memory storage for project analysis (replace with a database in production)
project_data_store: Dict[str, Any] = {}

# Use /tmp for temporary file storage on Vercel (writable directory)
TEMP_PROJECT_DIR = Path("/tmp") / "ai_assistant_projects"

# Predefined list of common OpenRouter models (user can select from these)
# The frontend will fetch this list to populate dropdowns.
OPENROUTER_MODELS = [
    "openai/gpt-3.5-turbo",
    "openai/gpt-4",
    "openai/gpt-4-turbo-preview", # Or specific preview/vision models
    "anthropic/claude-2",
    "anthropic/claude-3-opus", 
    "anthropic/claude-3-sonnet", 
    "anthropic/claude-3-haiku",
    "google/gemini-pro",
    "google/gemini-pro-vision",
    "mistralai/mistral-7b-instruct",
    "mistralai/mixtral-8x7b-instruct",
    "meta-llama/llama-2-70b-chat",
    "google/gemini-2.5-pro-preview"  # Suggested alternative from search
]

# --- Pydantic Models ---
class CodeGenerationRequest(BaseModel):
    prompt: str
    openrouter_model_name: str # Changed from model_identifier

class ProjectAnalysisRequest(BaseModel):
    openrouter_model_name: str

class PortfolioGenerationRequest(BaseModel):
    prompt: str
    openrouter_model_name: str

class ReactPortfolioRequest(BaseModel):
    prompt: str
    openrouter_model_name: str

# class CVGenerationRequest(BaseModel): # Old model, will be replaced
#     prompt: str
#     openrouter_model_name: str

# --- New Pydantic Models for Structured CV Data ---
class ExperienceEntry(BaseModel):
    job_title: str
    company_name: str
    location: Optional[str] = None
    start_date: str
    end_date: str # Can be "Present"
    description: str # Preferably bullet points, passed as a single string

class EducationEntry(BaseModel):
    degree_name: str
    institution_name: str
    graduation_date: str # Or expected date
    details: Optional[str] = None # e.g., GPA, relevant coursework, honors

class ProjectEntry(BaseModel):
    project_name: str
    description: str
    technologies_used: Optional[List[str]] = None
    link: Optional[str] = None

class CVDataRequest(BaseModel):
    full_name: str
    email: str
    phone: Optional[str] = None
    linkedin_url: Optional[str] = None
    github_url: Optional[str] = None
    website_url: Optional[str] = None
    professional_summary: str
    experiences: List[ExperienceEntry] = []
    educations: List[EducationEntry] = []
    projects: List[ProjectEntry] = []
    skills: List[str] = [] # List of skill strings
    is_fresher: Optional[bool] = False
# --- End New Pydantic Models ---

# --- Code Assistant Logic ---
class OpenRouterCodeAssistant:
    def __init__(self):
        self.openrouter_client = None
        
        openrouter_api_key_env = os.getenv("OPENROUTER_API_KEY")
        # The previously hardcoded key, kept here for reference if needed for a specific local fallback, but env var is primary.
        openrouter_api_key_hardcoded_fallback = "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"
        openrouter_base_url_env = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        
        api_key_to_use = None
        source_of_key = ""

        if openrouter_api_key_env:
            api_key_to_use = openrouter_api_key_env
            source_of_key = "environment variable OPENROUTER_API_KEY"
        elif openrouter_api_key_hardcoded_fallback: # Fallback only if env var is MISSING
            # This fallback is generally not recommended for production.
            # For Vercel, ensure OPENROUTER_API_KEY is set in project environment variables.
            api_key_to_use = openrouter_api_key_hardcoded_fallback
            source_of_key = "hardcoded fallback key"
            print(f"WARNING: Using {source_of_key}. For secure and proper configuration, please set the OPENROUTER_API_KEY environment variable.")
        
        if api_key_to_use:
            try:
                self.openrouter_client = openai.OpenAI(
                    api_key=api_key_to_use,
                    base_url=openrouter_base_url_env
                )
                print(f"OpenRouter client initialized using API key from {source_of_key} (Base URL: {openrouter_base_url_env}).")
            except Exception as e:
                print(f"CRITICAL: Failed to initialize OpenRouter client with key from {source_of_key}: {e}")
                self.openrouter_client = None # Ensure it's None on failure
        else:
            print("CRITICAL: OpenRouter API key not found in environment variables (OPENROUTER_API_KEY) or as a hardcoded fallback. API calls will fail.")
            self.openrouter_client = None
            
        self.supported_languages = {
            "python": {"extensions": [".py"]},
            "html": {"extensions": [".html", ".htm"]},
            "css": {"extensions": [".css"]},
            "javascript": {"extensions": [".js"]}
        }
        # Ensure TEMP_PROJECT_DIR exists
        TEMP_PROJECT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Temporary project uploads will be stored in: {TEMP_PROJECT_DIR.resolve()}")

        # Clean up old temporary project directories on startup - Commented out for Vercel
        # self._cleanup_old_projects()

    # def _cleanup_old_projects(self):
    #     # Simple cleanup: remove any directories in TEMP_PROJECT_DIR older than, say, 24 hours
    #     # For a more robust solution, track project creation times
    #     cutoff = 24 * 60 * 60  # 24 hours in seconds
    #     now = tempfile._get_candidate_names() # A bit of a hack to get a timestamp-like string for comparison logic
    #     # This cleanup is very basic. A real app would need better tracking or a scheduled job.
    #     try:
    #         for item in TEMP_PROJECT_DIR.iterdir():
    #             if item.is_dir():
    #                 try:
    #                     # Basic check, not a perfect age check
    #                     if item.stat().st_mtime < (float(now) - cutoff if now else 0): # Simplistic age check
    #                          shutil.rmtree(item)
    #                          print(f"Cleaned up old project directory: {item}")
    #                 except Exception as e:
    #                     print(f"Error during cleanup of {item}: {e}")
    #     except Exception as e:
    #         print(f"Error during initial cleanup scan: {e}")

    def _get_project_path(self, project_id: str) -> Path:
        # Sanitize project_id to prevent path traversal issues
        safe_project_id = Path(project_id).name
        return TEMP_PROJECT_DIR / safe_project_id

    def generate_code(self, prompt_text: str, model_name: str, language: str = "python") -> str:
        if not self.openrouter_client:
            return "# OpenRouter client not initialized. Check API key."
        
        system_message = f"You are an expert {language} programmer. Generate only the raw code as requested. Do not include explanations or markdown backticks around the code block. Just output the code."
        
        try:
            completion = self.openrouter_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt_text}
                ],
                temperature=0.7,
                max_tokens=2048
            )
            code = completion.choices[0].message.content.strip()
            # Attempt to remove markdown if present (though system prompt asks not to use it)
            if language != "markdown": # and language is not None: # Added check for language not None
                if code.startswith(f"```{language}") and code.endswith("```"):
                    code = code[len(f"```{language}"):-3].strip()
                elif code.startswith("```") and code.endswith("```"): # Generic ``` removal
                    code = code[3:-3].strip()
            return code
        except Exception as e:
            print(f"Error during OpenRouter code generation (model={model_name}): {e}")
            raise HTTPException(status_code=500, detail=f"Error generating code via OpenRouter: {e}")

    def generate_project_summary(self, technical_analysis: Dict[str, Any], model_name: str) -> str:
        if not self.openrouter_client:
            raise HTTPException(status_code=503, detail="OpenRouter client not configured for summary generation.")

        prompt_data = {
            "project_name": technical_analysis.get("project_name", "N/A"),
            "language": technical_analysis.get("language", "N/A"),
            "total_files": technical_analysis.get("overall_metrics", {}).get("total_files", 0),
            "total_sloc": technical_analysis.get("overall_metrics", {}).get("total_sloc", 0),
            "average_complexity": technical_analysis.get("overall_metrics", {}).get("average_complexity", 0),
            "average_maintainability_index": technical_analysis.get("overall_metrics", {}).get("average_maintainability_index", 0),
            "key_files_summary": [
                {"path": f.get("path"), "classes": len(f.get("classes",[])), "functions": len(f.get("functions",[]))} 
                for f in technical_analysis.get("files", [])[:5] # Summary of first 5 files
            ]
        }
        summary_prompt_text = f"""
        Analyze the following project's technical data and provide a concise, insightful summary (around 150-200 words).
        Focus on the project's structure, potential complexity, and key characteristics. Output only the summary text.

        Technical Data:
        {json.dumps(prompt_data, indent=2)}

        LLM Generated Summary:
        """
        try:
            completion = self.openrouter_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant skilled in summarizing software project data."},
                    {"role": "user", "content": summary_prompt_text}
                ],
                temperature=0.5,
                max_tokens=400
            )
            summary = completion.choices[0].message.content.strip()
            # Remove the "LLM Generated Summary:" prefix if it's there
            return summary.split("LLM Generated Summary:")[-1].strip() if "LLM Generated Summary:" in summary else summary
        except Exception as e:
            print(f"Error during OpenRouter summary generation (model={model_name}): {e}")
            raise HTTPException(status_code=500, detail=f"Error generating project summary via OpenRouter: {e}")

    def generate_static_portfolio(self, request: PortfolioGenerationRequest) -> Dict[str, str]:
        if not self.openrouter_client:
            raise HTTPException(status_code=503, detail="OpenRouter client not initialized. Check API key.")

        meta_prompt = f"""
        You are an expert full-stack web developer, specializing in creating stunning, modern, and interactive single-page personal web portfolios.
        Your task is to generate the complete HTML, CSS, and JavaScript for a portfolio based on the user's requirements.

        User Requirements: "{request.prompt}"

        **General Guidelines for Generation:**
        1.  **Modern & Visually Appealing Design**: The portfolio should have a clean, professional, and contemporary look and feel. Use good typography, spacing, and a cohesive color scheme. Aim for a design that would impress a potential employer or client in 2024.
        2.  **Responsive Design**: The CSS must ensure the portfolio is responsive and looks great on all devices (desktops, tablets, mobiles).
        3.  **Interactive Elements**: Incorporate subtle JavaScript-driven interactions or CSS animations/transitions to enhance user experience (e.g., smooth scrolling, hover effects, on-scroll reveal animations for sections).
        4.  **Standard Portfolio Sections**: Unless the user specifies otherwise, try to include common sections like:
            *   A compelling Hero/Header section.
            *   An 'About Me' section.
            *   A 'Projects' or 'Portfolio Showcase' section. This should be designed to flexibly accommodate images, videos, or even placeholders for interactive 3D models.
            *   A 'Skills' section (if relevant from the prompt).
            *   A 'Contact' section (e.g., with social media links or a simple contact form structure - no backend processing for the form).
        5.  **Placeholder for 3D Models**: If the user mentions 3D, design, or visual art, or if it seems appropriate for their field, include a dedicated section or an item within the project showcase clearly marked as a placeholder for a 3D model viewer. Add an HTML comment like `<!-- Placeholder for 3D Model Viewer (e.g., using Three.js, React Three Fiber, or an iframe embed) -->` within this section. The surrounding layout should be suitable for embedding an interactive 3D scene.
        6.  **Clean Code**: Ensure the generated HTML, CSS, and JavaScript are well-formatted, commented where necessary, and follow best practices.

        **Output Structure (Use these exact markers):**

        <<<FILE: index.html>>>
        [Complete and well-formed HTML5 content for index.html. It MUST link to `styles.css` and `script.js` appropriately (e.g., `<link rel="stylesheet" href="styles.css">` in `<head>` and `<script src="script.js" defer></script>` before `</body>`). Use semantic HTML5 tags.]
        <<<EOF>>>

        <<<FILE: styles.css>>>
        [Complete CSS content for styles.css. Use modern CSS techniques, including Flexbox or Grid for layout. Ensure styles are well-organized and commented if complex. Do not use CSS preprocessors like SASS/LESS.]
        <<<EOF>>>

        <<<FILE: script.js>>>
        [Complete JavaScript content for script.js. Use vanilla JavaScript. Add interactivity as described in the guidelines. If no complex JS is needed, a simple script for smooth scrolling or to handle a mobile navigation menu toggle would be appropriate. Ensure the script is non-blocking (`defer` attribute on script tag in HTML).]
        <<<EOF>>>

        Provide complete, runnable code for each file within its designated markers.
        """
        try:
            print(f"Generating rich static portfolio with model: {request.openrouter_model_name}")
            completion = self.openrouter_client.chat.completions.create(
                model=request.openrouter_model_name,
                messages=[
                    {"role": "system", "content": "You are an expert full-stack web developer that generates HTML, CSS, and JS for a portfolio site."},
                    {"role": "user", "content": meta_prompt}
                ],
                temperature=0.7, 
                max_tokens=3500
            )
            full_response = completion.choices[0].message.content
            
            files = {}
            expected_files = ["index.html", "styles.css", "script.js"]
            for filename in expected_files:
                start_marker = f"<<<FILE: {filename}>>>"
                end_marker = "<<<EOF>>>"
                
                start_index = full_response.find(start_marker)
                if start_index == -1:
                    files[filename] = f"/* Content for {filename} not found in LLM response. Ensure the LLM uses the markers correctly. */"
                    continue
                
                end_index = full_response.find(end_marker, start_index + len(start_marker))
                if end_index == -1:
                    file_content = full_response[start_index + len(start_marker):].strip()
                    if filename != "script.js": 
                         file_content += f"\n\n/* WARNING: End marker <<<EOF>>> missing for {filename}. File content might be incomplete. */"
                    files[filename] = file_content
                    continue
                
                file_content = full_response[start_index + len(start_marker) : end_index].strip()
                files[filename] = file_content
            
            if not files.get("index.html") or "not found in LLM response" in files.get("index.html", ""):
                 raise HTTPException(status_code=500, detail="LLM failed to generate index.html for the static portfolio. Please try a different prompt or model, or check the LLM's adherence to file markers.")

            return files

        except openai.APIError as e:
            print(f"OpenAI APIError during portfolio generation (model={request.openrouter_model_name}): {e}")
            raise HTTPException(status_code=e.status_code if hasattr(e, 'status_code') else 500, detail=f"OpenRouter API error: {e.message if hasattr(e, 'message') else str(e)}")
        except Exception as e:
            print(f"Generic error during portfolio generation (model={request.openrouter_model_name}): {e}")
            raise HTTPException(status_code=500, detail=f"Error generating portfolio: {e}")

    def generate_react_portfolio(self, request: ReactPortfolioRequest) -> Dict[str, str]:
        if not self.openrouter_client:
            raise HTTPException(status_code=503, detail="OpenRouter client not initialized. Check API key.")

        meta_prompt = f"""
        You are an expert React.js developer tasked with scaffolding a modern, single-page personal web portfolio application.
        Generate the necessary file structure and code for a basic React application based on the user's requirements.

        User Requirements: "{request.prompt}"

        **Core React Project Structure to Generate:**
        - `public/index.html`: The main HTML page for the React app.
        - `src/index.js`: The entry point that renders the React app.
        - `src/App.js`: The main application component that routes or lays out other components.
        - `src/App.css`: Global styles for the application.
        - `src/components/`: A directory for reusable UI components.

        **Component Breakdown (Suggest common components, adapt to user prompt):**
        - `src/components/Header.js`: Navigation bar / hero section.
        - `src/components/About.js`: About me section.
        - `src/components/Projects.js`: Section to display a list/grid of projects.
        - `src/components/ProjectCard.js`: Component for an individual project item.
        - `src/components/Contact.js`: Contact form / information section.
        - `src/components/ThreeDPlaceholder.js`: (If 3D is mentioned or implied) A placeholder component where a 3D model viewer could be integrated later. It should render a simple div with a message like '3D Model Will Be Displayed Here'.

        **Guidelines for Generation:**
        1.  **React Functional Components**: Use functional components with hooks (e.g., `useState`, `useEffect`). Avoid class components.
        2.  **Basic Styling**: Provide basic CSS in `src/App.css` to make the application presentable. Aim for a clean, modern look. Simple Flexbox or Grid layouts are encouraged.
        3.  **Imports/Exports**: Ensure all components are correctly exported and imported where used.
        4.  **Placeholder Content**: Use placeholder text (Lorem Ipsum) and image placeholders (e.g., from https://via.placeholder.com/300) if specific content isn't derived from the user's prompt.
        5.  **No Build Configuration**: Do NOT generate `package.json`, `webpack.config.js`, or other build tool configurations. The output should be the source files that a developer would place into an existing React project setup (e.g., one created with Create React App or Vite).
        6.  **Clean Code**: Well-formatted JSX and JavaScript. Add comments for clarity if needed.

        **Output Structure (Use these exact markers for each file):**

        <<<FILE: public/index.html>>>
        [HTML content for public/index.html. This is the standard shell for a React app, typically with a `<div id="root"></div>`.]
        <<<EOF>>>

        <<<FILE: src/index.js>>>
        [JavaScript content for src/index.js. Imports React, ReactDOM, App, and App.css, then renders `<App />` into the root div.]
        <<<EOF>>>

        <<<FILE: src/App.js>>>
        [JSX content for src/App.js. Imports necessary components and lays out the main structure of the portfolio page.]
        <<<EOF>>>

        <<<FILE: src/App.css>>>
        [CSS content for src/App.css. Basic global styles and styles for App.js layout.]
        <<<EOF>>>

        <<<FILE: src/components/Header.js>>>
        [JSX content for Header component.]
        <<<EOF>>>

        <<<FILE: src/components/About.js>>>
        [JSX content for About component.]
        <<<EOF>>>
        
        <<<FILE: src/components/Projects.js>>>
        [JSX content for Projects component. Should import and use ProjectCard.]
        <<<EOF>>>

        <<<FILE: src/components/ProjectCard.js>>>
        [JSX content for ProjectCard component.]
        <<<EOF>>>

        <<<FILE: src/components/Contact.js>>>
        [JSX content for Contact component.]
        <<<EOF>>>

        (Include ThreeDPlaceholder.js if relevant, following the same marker pattern)
        <<<FILE: src/components/ThreeDPlaceholder.js>>>
        [JSX content for ThreeDPlaceholder component.]
        <<<EOF>>>

        Provide complete, runnable (within a React project) code for each file.
        Ensure all JavaScript/JSX files that are React components start with `import React from 'react';`
        """
        try:
            print(f"Generating React portfolio structure with model: {request.openrouter_model_name}")
            completion = self.openrouter_client.chat.completions.create(
                model=request.openrouter_model_name,
                messages=[
                    {"role": "system", "content": "You are an AI assistant that generates React.js project source code files based on user requirements and a predefined structure."},
                    {"role": "user", "content": meta_prompt}
                ],
                temperature=0.6, # Slightly lower temp for more structured code
                max_tokens=4000  # Needs to be high for multiple complex files
            )
            full_response = completion.choices[0].message.content

            files = {}
            # Define a more flexible way to list expected files based on prompt or LLM convention
            # For now, list common ones. The LLM is instructed to create them.
            # Parsing needs to be robust to handle variations in which component files are generated.
            # We'll search for any <<<FILE: ...>>> marker.

            current_pos = 0
            while True:
                start_marker_begin = full_response.find("<<<FILE: ", current_pos)
                if start_marker_begin == -1:
                    break # No more file markers found
                
                start_marker_end = full_response.find(">>>", start_marker_begin)
                if start_marker_end == -1:
                    break # Malformed start marker
                
                filename = full_response[start_marker_begin + len("<<<FILE: "):start_marker_end].strip()
                content_start_index = start_marker_end + len(">>>")
                
                end_marker_search_pos = content_start_index
                end_marker_index = full_response.find("<<<EOF>>>", end_marker_search_pos)
                
                if end_marker_index == -1:
                    # If EOF is missing, take rest of content for this file, warn, and stop.
                    print(f"Warning: End marker <<<EOF>>> missing for {filename} after position {content_start_index}. Taking remaining content.")
                    file_content = full_response[content_start_index:].strip()
                    if file_content: # Only add if there is some content
                        file_content += f"\n\n/* WARNING: End marker <<<EOF>>> was missing. This file ({filename}) or subsequent files might be incomplete. */"
                        files[filename] = file_content
                    break # Stop processing further files as structure is broken
                
                file_content = full_response[content_start_index:end_marker_index].strip()
                files[filename] = file_content
                current_pos = end_marker_index + len("<<<EOF>>>")
            
            # Basic check for essential files
            essential_files = ["public/index.html", "src/index.js", "src/App.js"]
            missing_essentials = [f for f in essential_files if f not in files or not files[f].strip()]
            if missing_essentials:
                raise HTTPException(status_code=500, detail=f"LLM failed to generate essential React files: {', '.join(missing_essentials)}. Please check LLM output or try a different prompt/model.")

            return files

        except openai.APIError as e:
            print(f"OpenAI APIError during React portfolio generation (model={request.openrouter_model_name}): {e}")
            raise HTTPException(status_code=e.status_code if hasattr(e, 'status_code') else 500, detail=f"OpenRouter API error: {e.message if hasattr(e, 'message') else str(e)}")
        except Exception as e:
            print(f"Generic error during React portfolio generation (model={request.openrouter_model_name}): {e}")
            raise HTTPException(status_code=500, detail=f"Error generating React portfolio: {str(e)}")

    def generate_cv_html(self, request: CVDataRequest) -> str:
        if not self.openrouter_client:
            raise HTTPException(status_code=503, detail="OpenRouter client not initialized. Check API key.")

        fixed_cv_model = "google/gemini-2.5-pro-preview" # Fixed model for CV generation

        # Construct the detailed user data section for the prompt
        user_data_details = f"Full Name: {request.full_name}\n"
        user_data_details += f"Email: {request.email}\n"
        if request.phone: user_data_details += f"Phone: {request.phone}\n"
        if request.linkedin_url: user_data_details += f"LinkedIn: {request.linkedin_url}\n"
        if request.github_url: user_data_details += f"GitHub: {request.github_url}\n"
        if request.website_url: user_data_details += f"Website: {request.website_url}\n"
        user_data_details += f"\nProfessional Summary:\n{request.professional_summary}\n"
        
        if request.is_fresher:
            user_data_details += "\nUser is a fresher (no prior work experience).\n"
        elif request.experiences: # Only add experience if not fresher AND experiences exist
            user_data_details += "\nWork Experience:\n"
            for exp in request.experiences:
                user_data_details += f"  - Job Title: {exp.job_title}\n"
                user_data_details += f"    Company: {exp.company_name}{f', {exp.location}' if exp.location else ''}\n"
                user_data_details += f"    Dates: {exp.start_date} - {exp.end_date}\n"
                user_data_details += f"    Description:\n      {exp.description.replace('\n', '\n      ')}\n"
        
        if request.educations:
            user_data_details += "\nEducation:\n"
            for edu in request.educations:
                user_data_details += f"  - Degree: {edu.degree_name}\n"
                user_data_details += f"    Institution: {edu.institution_name}\n"
                user_data_details += f"    Graduation Date: {edu.graduation_date}\n"
                if edu.details: user_data_details += f"    Details: {edu.details}\n"

        if request.projects:
            user_data_details += "\nProjects:\n"
            for proj in request.projects:
                user_data_details += f"  - Project Name: {proj.project_name}\n"
                user_data_details += f"    Description:\n      {proj.description.replace('\n', '\n      ')}\n"
                if proj.technologies_used: user_data_details += f"    Technologies: {', '.join(proj.technologies_used)}\n"
                if proj.link: user_data_details += f"    Link: {proj.link}\n"

        if request.skills:
            user_data_details += "\nSkills:\n  - " + ", ".join(request.skills) + "\n"
        
        fresher_guidance = ""
        if request.is_fresher:
            fresher_guidance = "Since the user is a fresher, focus primarily on their Education, Projects, and Skills. The Work Experience section should be omitted or very minimal if any placeholder is absolutely needed (prefer omission)."

        meta_prompt = f"""
        You are an expert CV and resume designer. Your task is to generate a complete, professional, single HTML file for a CV based on the structured user data provided below.
        The CV should be designed to look good when printed on A4 paper. Use the provided data to populate the relevant sections of the CV.
        {fresher_guidance}

        **Structured User Data:**
        {user_data_details}

        # This was in the original cv generation and it contains the user inputs such as name email and etc and it is needed for this cv generation

        **Key Requirements for the HTML CV:**
        1.  **Single HTML File**: All CSS must be embedded within `<style>` tags in the `<head>` section of the HTML. Do not link to external CSS files.
        2.  **A4 Paper Size Simulation**: The CSS should style the main CV container to approximate A4 dimensions. Example CSS to guide you (adapt and expand as needed):
            ```css
            body {{
                background-color: #f0f0f0; /* So the A4 page stands out */
                display: flex;
                justify-content: center;
                align-items: flex-start; /* Align page to top */
                padding: 20px 0;
                margin: 0;
                font-family: 'Arial', sans-serif; /* Or another professional font like 'Lato', 'Open Sans' */
                line-height: 1.6;
            }}
            .cv-container {{
                background-color: #fff;
                width: 210mm;
                min-height: 290mm; /* Slightly less than 297mm to allow for browser differences, content drives height */
                padding: 15mm;
                border: 1px solid #ccc;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
                box-sizing: border-box;
            }}
            h1, h2, h3, h4 {{ margin-top: 0.5em; margin-bottom: 0.3em; color: #333; }}
            h1 {{ font-size: 24pt; text-align: center; border-bottom: 2px solid #eee; padding-bottom: 0.2em; margin-bottom: 0.5em; }}
            h2 {{ font-size: 16pt; border-bottom: 1px solid #eee; padding-bottom: 0.1em; margin-top: 1em; }}
            p {{ margin: 0.2em 0; }}
            ul {{ padding-left: 20px; margin: 0.2em 0; }}
            li {{ margin-bottom: 0.1em; }}
            .contact-info p {{ margin: 0; font-size: 10pt; }}
            .section {{ margin-bottom: 1em; }}
            .job, .education-entry, .project-entry {{ margin-bottom: 0.8em; }}
            .job-title, .degree-name, .project-name {{ font-weight: bold; font-size: 12pt; color: #444; }}
            .company-name, .institution-name {{ font-style: italic; font-size: 11pt; color: #555; }}
            .dates {{ float: right; font-size: 10pt; color: #666; }}
            .description ul li {{ font-size: 10pt; }}
            /* Add more professional styling for overall look and feel. Consider good typography and spacing. */
            ```
        3.  **Professional Design**: Clean, modern, readable layout. Ensure good visual hierarchy.
        4.  **Standard CV Sections**: Use the provided structured data to create sections like Contact Information, Summary, Work Experience (if applicable and data provided), Education, Projects, Skills. Format them professionally.
        5.  **HTML Structure**: Use semantic HTML5 tags (`<header>`, `<section>`, `<article>`, `<h2>`, etc.).
        6.  **No JavaScript**: Output pure HTML and CSS.

        **Output directly the complete HTML content. Do not wrap it in markdown backticks or provide any explanations outside the HTML itself.**
        Start with `<!DOCTYPE html><html><head>...</head><body><div class="cv-container">...</div></body></html>`
        """
        try:
            print(f"Generating CV HTML with fixed model: {fixed_cv_model} using structured data.")
            completion = self.openrouter_client.chat.completions.create(
                model=fixed_cv_model, # Use the fixed model
                messages=[
                    {"role": "system", "content": "You are an AI assistant that generates a complete, professional, single HTML file for a CV, styled for A4 paper, using provided structured data."},
                    {"role": "user", "content": meta_prompt}
                ],
                temperature=0.5,
                max_tokens=4000
            )
            cv_html = completion.choices[0].message.content.strip()
            
            # Rudimentary check for HTML structure
            if not (cv_html.lower().startswith("<!doctype html") or cv_html.lower().startswith("<html>")):
                if "```html" in cv_html and "```" in cv_html.split("```html")[1]:
                    cv_html = cv_html.split("```html")[1].split("```")[0].strip()
                elif cv_html.startswith("```") and cv_html.endswith("```"):
                     cv_html = cv_html[3:-3].strip()
                else:
                    print(f"Warning: CV generation response for model {fixed_cv_model} did not start with HTML doctype as expected. Returning raw response. Check LLM output for issues.")

            return cv_html

        except openai.APIError as e:
            print(f"OpenAI APIError during CV generation (model={fixed_cv_model}): {e}")
            raise HTTPException(status_code=e.status_code if hasattr(e, 'status_code') else 500, detail=f"OpenRouter API error during CV generation: {e.message if hasattr(e, 'message') else str(e)}")
        except Exception as e:
            print(f"Generic error during CV generation (model={fixed_cv_model}): {e}")
            raise HTTPException(status_code=500, detail=f"Error generating CV: {str(e)}")

    # --- Project Analysis Methods (largely unchanged but ensure they use Path objects) ---
    def _is_relevant_file(self, file_path: Path, language: str) -> bool:
        lang_config = self.supported_languages.get(language.lower())
        if not lang_config:
            return False
        return file_path.suffix.lower() in lang_config.get("extensions", [])

    def _analyze_python_file(self, file_path: Path) -> Dict[str, Any]:
        analysis: Dict[str, Any] = {
            "path": str(file_path.relative_to(file_path.parent.parent if file_path.parent.name == 'extracted' else file_path.parent)), # Make path relative to project root
            "size": file_path.stat().st_size,
            "sloc": 0, "lloc": 0, "comments":0, "blank_lines": 0,
            "cyclomatic_complexity": [], "maintainability_index": "N/A",
            "functions": [], "classes": [], "imports": []
        }
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            if not content.strip(): # Skip empty files for radon analysis
                analysis["error"] = "File is empty."
                return analysis

            raw_metrics = raw_analyze(content)
            analysis["sloc"] = raw_metrics.sloc
            analysis["lloc"] = raw_metrics.lloc
            analysis["comments"] = raw_metrics.comments + raw_metrics.multi # sum of single and multi-line comments
            analysis["blank_lines"] = raw_metrics.blank
            
            cc_results = cc_visit(content)
            for item in cc_results:
                analysis["cyclomatic_complexity"].append({
                    "name": item.name, "type": item.type,
                    "complexity": item.complexity, "rank": item.rank()
                })
            
            # Radon's mi_visit can fail on very small or syntactically incomplete snippets
            try:
                analysis["maintainability_index"] = round(mi_visit(content, multi=True), 2)
            except Exception: # Catch specific radon errors if known, otherwise generic
                analysis["maintainability_index"] = "Error"

            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    analysis["functions"].append({
                        "name": node.name,
                        "args": [arg.arg for arg in node.args.args],
                        "docstring_present": bool(ast.get_docstring(node))
                    })
                elif isinstance(node, ast.ClassDef):
                    analysis["classes"].append({
                        "name": node.name,
                        "methods_count": len([m for m in node.body if isinstance(m, ast.FunctionDef)]),
                        "docstring_present": bool(ast.get_docstring(node))
                    })
                elif isinstance(node, ast.Import):
                    analysis["imports"].extend([n.name for n in node.names])
                elif isinstance(node, ast.ImportFrom):
                    module_name = node.module or "." # Handle relative imports from current dir
                    analysis["imports"].extend([f"{module_name}.{n.name}" for n in node.names])
        except SyntaxError as se:
            analysis["error"] = f"Syntax error: {se.msg} at line {se.lineno}"
        except Exception as e:
            analysis["error"] = f"Error analyzing file: {str(e)}"
        return analysis

    def analyze_project_files(self, project_id: str, language: str = "python") -> Dict[str, Any]:
        project_path_container = self._get_project_path(project_id)
        
        # Determine the actual root of the extracted project content
        # Common case: project.zip extracts to project_path_container/extracted/project_files...
        # Or: project.zip extracts to project_path_container/project_folder/project_files...
        # Or: project.zip extracts directly to project_path_container/project_files... (no 'extracted' or single top-level folder)
        
        extracted_content_root = project_path_container / "extracted"
        if not extracted_content_root.exists() or not extracted_content_root.is_dir():
            # If 'extracted' doesn't exist, assume files are directly in project_path_container or in one subfolder
            sub_items = list(project_path_container.iterdir())
            # If there's exactly one subdirectory, assume that's the project root
            if len(sub_items) == 1 and sub_items[0].is_dir():
                extracted_content_root = sub_items[0]
            else: # Otherwise, assume files are directly in project_path_container
                extracted_content_root = project_path_container
        
        if not extracted_content_root.exists() or not extracted_content_root.is_dir():
            raise ValueError(f"Could not locate extracted project content for '{project_id}'. Expected in '{project_path_container / 'extracted'}' or similar.")

        project_name_for_display = extracted_content_root.name
        if project_name_for_display == "extracted":
             # Try to get a more meaningful name if possible
             potential_project_dirs = [d for d in extracted_content_root.iterdir() if d.is_dir()]
             if len(potential_project_dirs) == 1:
                 project_name_for_display = potential_project_dirs[0].name
             # else, stick with 'extracted' or project_id

        project_analysis: Dict[str, Any] = {
            "project_id": project_id,
            "project_name": project_name_for_display,
            "scanned_path": str(extracted_content_root.resolve()),
            "language": language,
            "files": [],
            "overall_metrics": {
                "total_files_scanned": 0, # All files encountered
                "total_relevant_files_analyzed": 0, # Files matching language criteria
                "total_sloc": 0,
                "total_lloc": 0,
                "total_comments": 0,
                "total_blank_lines": 0,
                "average_complexity": 0,
                "average_maintainability_index": 0,
                "files_with_errors": 0
            }
        }
        total_complexity_sum = 0
        total_mi_sum = 0
        analyzed_files_count_for_mi = 0
        analyzed_functions_for_cc = 0

        for item_path in extracted_content_root.rglob("*"):
            project_analysis["overall_metrics"]["total_files_scanned"] += 1
            if item_path.is_file() and self._is_relevant_file(item_path, language):
                project_analysis["overall_metrics"]["total_relevant_files_analyzed"] += 1
                if language.lower() == "python":
                    file_analysis = self._analyze_python_file(item_path)
                    project_analysis["files"].append(file_analysis)
                    
                    if "error" in file_analysis:
                        project_analysis["overall_metrics"]["files_with_errors"] += 1
                    else:
                        project_analysis["overall_metrics"]["total_sloc"] += file_analysis.get("sloc", 0)
                        project_analysis["overall_metrics"]["total_lloc"] += file_analysis.get("lloc", 0)
                        project_analysis["overall_metrics"]["total_comments"] += file_analysis.get("comments",0)
                        project_analysis["overall_metrics"]["total_blank_lines"] += file_analysis.get("blank_lines",0)

                        if isinstance(file_analysis.get("maintainability_index"), (int, float)):
                            total_mi_sum += file_analysis["maintainability_index"]
                            analyzed_files_count_for_mi += 1
                        
                        for cc_item in file_analysis.get("cyclomatic_complexity", []):
                            total_complexity_sum += cc_item["complexity"]
                            analyzed_functions_for_cc +=1
                # Add elif for other languages here if needed for technical analysis (e.g., JS using esprima)
            
        if analyzed_functions_for_cc > 0:
            project_analysis["overall_metrics"]["average_complexity"] = round(total_complexity_sum / analyzed_functions_for_cc, 2)
        if analyzed_files_count_for_mi > 0:
            project_analysis["overall_metrics"]["average_maintainability_index"] = round(total_mi_sum / analyzed_files_count_for_mi, 2)

        return project_analysis

    def save_uploaded_project(self, project_id: str, file: UploadFile) -> Path:
        # Ensure project_id is just a name, not a path segment
        safe_project_id = Path(project_id).name 
        project_path_container = self._get_project_path(safe_project_id) # Use the sanitized ID

        # Clean up if directory already exists for this ID to prevent conflicts
        if project_path_container.exists():
            shutil.rmtree(project_path_container)
        project_path_container.mkdir(parents=True, exist_ok=True)
        
        # Sanitize filename from UploadFile object
        safe_filename = Path(file.filename).name if file.filename else "project.zip"
        zip_file_path = project_path_container / safe_filename
        
        try:
            with open(zip_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            extracted_dir = project_path_container / "extracted"
            # No need to check for extracted_dir existence here as we cleaned the parent
            extracted_dir.mkdir(exist_ok=True) 
            
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(extracted_dir)
            
            return extracted_dir # Return path to the directory where files are extracted
        except zipfile.BadZipFile:
            shutil.rmtree(project_path_container) # Clean up on bad zip
            raise HTTPException(status_code=400, detail="Invalid ZIP file.")
        except Exception as e:
            shutil.rmtree(project_path_container) # Clean up on other errors
            raise HTTPException(status_code=500, detail=f"Failed to save or extract project: {e}")
        finally:
            if zip_file_path.exists():
                os.remove(zip_file_path) # Clean up the zip file itself after extraction or error

assistant = OpenRouterCodeAssistant()

# --- API Endpoints ---
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def get_index():
    try:
        with open("templates/index.html", "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Index.html not found in templates directory.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading index.html: {e}")

@app.get("/api/openrouter-models", response_model=List[str])
async def get_openrouter_models():
    """Returns the predefined list of OpenRouter model strings."""
    return OPENROUTER_MODELS

@app.post("/api/generate-code", summary="Generate Code Snippet")
async def generate_code_endpoint(request: CodeGenerationRequest):
    """Generates a code snippet based on a prompt using a specified OpenRouter model."""
    try:
        # Assuming language is part of the prompt or a fixed default for now (e.g., Python)
        # If language needs to be dynamic, it should be part of CodeGenerationRequest
        generated_code = assistant.generate_code(
            prompt_text=request.prompt, 
            model_name=request.openrouter_model_name
            # language=request.language # If language is added to request
        )
        return {"generated_code": generated_code, "openrouter_model_name": request.openrouter_model_name}
    except HTTPException as e:
        raise e # Re-raise HTTPException to preserve status code and detail
    except Exception as e:
        print(f"Unexpected error in /api/generate-code: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.post("/api/upload-and-analyze-project", summary="Upload, Analyze Project, and Generate LLM Summary")
async def upload_and_analyze_project_endpoint(
    file: UploadFile = File(..., description="Project .zip file"),
    openrouter_model_name: str = Query(..., description="OpenRouter model identifier for summary generation (e.g., 'google/gemini-pro')")
):
    """
    Uploads a project as a .zip file, performs technical analysis (Python files only),
    and generates an LLM-based summary of the project.
    A unique `project_id` will be generated based on the filename.
    """
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a .zip file.")
    
    # Generate a simple project_id from the filename
    project_id = Path(file.filename).stem 

    try:
        assistant.save_uploaded_project(project_id, file) # This now returns path to extracted dir
        # Assuming Python for now, make language configurable if needed
        technical_analysis = assistant.analyze_project_files(project_id, language="python")
        llm_summary = assistant.generate_project_summary(technical_analysis, openrouter_model_name)
        
        return {
            "project_id": project_id,
            "technical_analysis": technical_analysis,
            "llm_summary": llm_summary,
            "summary_model_used": openrouter_model_name
        }
    except ValueError as ve: # Catch specific errors like project not found
        raise HTTPException(status_code=404, detail=str(ve))
    except HTTPException as e:
        raise e # Re-raise if it's already an HTTPException (e.g. from save_uploaded_project)
    except Exception as e:
        print(f"Error in upload/analysis for project {project_id}: {e}")
        # Attempt to clean up the project directory on error
        try:
            project_dir_to_clean = assistant._get_project_path(project_id)
            if project_dir_to_clean.exists():
                shutil.rmtree(project_dir_to_clean)
        except Exception as cleanup_e:
            print(f"Error during cleanup for project {project_id}: {cleanup_e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during project processing: {str(e)}")

@app.post("/api/generate-portfolio", summary="Generate Static Web Portfolio")
async def generate_portfolio_endpoint(request: PortfolioGenerationRequest):
    """
    Generates a static web portfolio (HTML, CSS, JS) based on a prompt
    and returns it as a downloadable ZIP file.
    """
    try:
        files_content = assistant.generate_static_portfolio(request)
        
        zip_io = io.BytesIO()
        with zipfile.ZipFile(zip_io, mode="w", compression=zipfile.ZIP_DEFLATED) as temp_zip:
            for filename, content in files_content.items():
                temp_zip.writestr(filename, content)
        
        zip_io.seek(0) # Rewind the buffer to the beginning
        
        return StreamingResponse(
            zip_io,
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=portfolio.zip"}
        )
    except HTTPException as e:
        raise e # Re-raise HTTPException to preserve status code and detail
    except Exception as e:
        print(f"Unexpected error in /api/generate-portfolio: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred while generating portfolio: {str(e)}")

@app.post("/api/generate-react-portfolio", summary="Generate React Portfolio Source Code")
async def generate_react_portfolio_endpoint(request: ReactPortfolioRequest):
    """
    Generates the source code structure for a React-based web portfolio 
    (JSX components, CSS, public/index.html, src/index.js) based on a prompt.
    Returns a downloadable ZIP file containing these source files.
    NOTE: This provides source code, not a built/runnable application directly.
    The user needs to set up a React project (e.g., with Create React App or Vite)
    and integrate these generated files.
    """
    try:
        files_content = assistant.generate_react_portfolio(request)
        
        zip_io = io.BytesIO()
        with zipfile.ZipFile(zip_io, mode="w", compression=zipfile.ZIP_DEFLATED) as temp_zip:
            for filename, content in files_content.items():
                # Ensure filenames are valid for zip (e.g., no leading slashes if not intended)
                # Path objects can help normalize, but here we assume LLM gives relative paths like 'src/App.js'
                temp_zip.writestr(filename, content)
        
        zip_io.seek(0)
        
        return StreamingResponse(
            zip_io,
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=react_portfolio_src.zip"}
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Unexpected error in /api/generate-react-portfolio: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred while generating React portfolio: {str(e)}")

@app.post("/api/generate-cv", summary="Generate CV from Prompt")
async def generate_cv_endpoint(request: CVDataRequest):
    """
    Generates a professional CV in HTML format based on a user prompt.
    The HTML includes embedded CSS styled for A4 paper.
    """
    try:
        # The assistant object is already instantiated globally in this file
        # For Vercel, instantiating per request or globally needs consideration for state.
        # Global `assistant = OpenRouterCodeAssistant()` should be fine for Vercel's stateless model.
        cv_html_output = assistant.generate_cv_html(request)
        return HTMLResponse(content=cv_html_output)
    except HTTPException as e:
        raise e # Re-raise HTTPException to preserve status code and detail
    except Exception as e:
        print(f"Unexpected error in /api/generate-cv: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred while generating CV: {str(e)}")

# The following block will be removed for Vercel deployment:
# if __name__ == "__main__":
#     # Ensure .env file is created if it doesn't exist for local dev
#     if not os.path.exists(".env"):
#         with open(".env", "w") as f:
#             f.write("OPENROUTER_API_KEY=your_openrouter_api_key_here\n")
#             f.write("OPENROUTER_BASE_URL=https://openrouter.ai/api/v1\n")
#         print("Created a default .env file. Please ensure OPENROUTER_API_KEY is set.")
#     else:
#         # Check if OPENROUTER_API_KEY is set if .env exists
#         load_dotenv() # Load again to ensure current values
#         if not os.getenv("OPENROUTER_API_KEY"):
#             print("Warning: .env file exists but OPENROUTER_API_KEY is not set or is empty.")
#         if not os.getenv("OPENROUTER_BASE_URL"):
#             # This is less critical if the default is used in the class, but good to note
#             print("Warning: OPENROUTER_BASE_URL is not set in .env. Defaulting to https://openrouter.ai/api/v1")


#     uvicorn.run(
#         app,
#         host=os.getenv("HOST", "127.0.0.1"),
#         port=int(os.getenv("PORT", 8000)),
#         reload=True # Enable reload for development
#     ) 