import streamlit as st
import pdf2image
import io
import json
import base64
import google.generativeai as genai
from openai import OpenAI # Import OpenAI library

# --- Configuration and Model Setup ---

# Initialize API keys from Streamlit secrets
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")

# --- AI Model Selection in Sidebar ---
st.sidebar.subheader("AI Model Selection")
ai_model_choice = st.sidebar.radio(
    "Choose your AI Model:",
    ('Google Gemini', 'OpenAI')
)

# Initialize AI models based on choice and available keys
if ai_model_choice == 'Google Gemini':
    if GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)
        # Using Gemini-2.0-flash as it supports multimodal input directly
        ai_model = genai.GenerativeModel('gemini-2.0-flash')
    else:
        st.warning("Google Gemini API Key not found in Streamlit secrets. Please add 'GOOGLE_API_KEY'.")
        ai_model = None
elif ai_model_choice == 'OpenAI':
    if OPENAI_API_KEY:
        try:
            openai_client = OpenAI(api_key=OPENAI_API_KEY)
            # OpenAI's multimodal model is gpt-4o, gpt-4o-mini
            # For text-only, you can use gpt-3.5-turbo or gpt-4-turbo
            ai_model = openai_client
        except Exception as e:
            st.error(f"Error initializing OpenAI client: {e}. Check your OpenAI API Key.")
            ai_model = None
    else:
        st.warning("OpenAI API Key not found in Streamlit secrets. Please add 'OPENAI_API_KEY'.")
        ai_model = None

# --- Generic AI Response Functions ---

@st.cache_data()
def get_ai_response(model_choice, _ai_model_instance, input_prompt, pdf_content, user_input, additional_params=None):
    """
    Generates a text response from the selected AI model (Gemini or OpenAI).
    """
    full_prompt = input_prompt
    if additional_params:
        for key, value in additional_params.items():
            full_prompt += f"\n\n--- {key} ---\n{value}"

    if model_choice == 'Google Gemini':
        if _ai_model_instance: # Use the underscore here too
            response = _ai_model_instance.generate_content([full_prompt, pdf_content[0], user_input])
            return response.text
        else:
            st.error("Gemini model not initialized. Please check API key.")
            return "Error: Gemini model not available."
    elif model_choice == 'OpenAI':
        if _ai_model_instance: # And here
            # For OpenAI's Vision models (like gpt-4o), you need to encode image data
            # OpenAI's chat completions API expects messages in a specific format
            # Convert PDF image to base64 for OpenAI Vision models
            image_b64 = pdf_content[0]["data"] # Assumes pdf_content[0] is the base64 image dict

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": full_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                        {"type": "text", "text": user_input}
                    ]
                }
            ]
            try:
                completion = _ai_model_instance.chat.completions.create( # And here
                    model="gpt-4o", # or "gpt-4o-mini"
                    messages=messages,
                    max_tokens=2000 # Adjust as needed
                )
                return completion.choices[0].message.content
            except Exception as e:
                st.error(f"Error calling OpenAI API: {e}")
                return "Error: OpenAI model not available or encountered an issue."
        else:
            st.error("OpenAI model not initialized. Please check API key.")
            return "Error: OpenAI model not available."
    return "Error: No AI model selected or initialized."

@st.cache_data()
def get_ai_response_keywords(model_choice, _ai_model_instance, input_prompt, pdf_content, user_input, additional_params=None):
    """
    Generates a JSON response with keywords from the selected AI model.
    """
    full_prompt = input_prompt
    if additional_params:
        for key, value in additional_params.items():
            full_prompt += f"\n\n--- {key} ---\n{value}"

    # Note: We pass _ai_model_instance here as well
    response_text = get_ai_response(model_choice, _ai_model_instance, input_prompt, pdf_content, user_input, additional_params)

    try:
        json_string = response_text.strip()
        if json_string.startswith('```json'):
            json_string = json_string[len('```json'):]
        if json_string.endswith('```'):
            json_string = json_string[:-len('```')]
        return json.loads(json_string.strip())
    except json.JSONDecodeError:
        st.error("Error parsing JSON response from the model. Please try again or refine the prompt.")
        st.write("Raw response for debugging:")
        st.code(response_text)
        return {}

@st.cache_data()
def input_pdf_setup(uploaded_file):
    """
    Processes the uploaded PDF file to extract the first page as a JPEG image.
    """
    if uploaded_file is not None:
        images = pdf2image.convert_from_bytes(uploaded_file.read())
        if images:
            first_page = images[0]
            img_byte_arr = io.BytesIO()
            first_page.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()
            pdf_parts = [
                {
                    "mime_type": "image/jpeg",
                    "data": base64.b64encode(img_byte_arr).decode()
                }
            ]
            return pdf_parts
        else:
            raise ValueError("Could not convert PDF to image. The file might be corrupted or empty.")
    else:
        raise FileNotFoundError("No file uploaded")

# --- Streamlit App UI ---

st.set_page_config(page_title="LSM ATS Resume Scanner")
st.header("LSM Application Tracking System")

# --- User Inputs ---
input_text = st.text_area("Job Description: ", key="input", height=200, help="Paste the full job description here.")
uploaded_file = st.file_uploader("Upload your resume (PDF)...", type=["pdf"])

if 'resume' not in st.session_state:
    st.session_state.resume = None

if uploaded_file is not None:
    st.write("PDF Uploaded Successfully")
    st.session_state.resume = uploaded_file

# --- Additional Parameters for Prompts (Checklists) ---
st.sidebar.header("LSM Prompt Customization")

# General Resume Checklist
general_resume_checklist = """
**General Format Checklist:**
- Used Microsoft Word (avoid templates for ATS compatibility)?
- Margins consistent (0.5-1.0 inches)?
- Font size 10-12 pt, easy to read (Arial, Calibri, Times New Roman)?
- One page (or two if 10+ years experience/advanced degree)?
- Enough white space?
- Bold/italics used appropriately (headers/positions), no underlining?
- Dates clear and consistent?
- Format and punctuation consistent?
- Sections in order of importance?
- Heading names descriptive (e.g., Research Experience, Leadership & Service)?

**Contact Information Checklist:**
- Legal name clear and bold at top (and second page)?
- Phone number included (professional voicemail)?
- Professional email address included?
- US citizen/permanent resident visa included if applicable?

**Education Checklist:**
- Institution names spelled out (e.g., Massachusetts Institute of Technology)?
- Official degree/course names listed?
- Month and year of degree earned/expected?
- GPA listed if strong (with scale)?
- Coursework aligning with job search listed?

**Experience Checklist:**
- Organization/company name and job title clear?
- City and state (or country) included?
- Employment dates listed for each role?
- Project, activity, and results listed for each experience?
- Each phrase starts with an action verb (past tense for previous, present for current)?
- Quantified relevant information (size, scale, budget, staff)?
- Keywords applied to industry/job listings?
- Avoided first-person pronouns (I, me, mine, myself)?
- All relevant experiences related to job opening(s) included?

**Skills Checklist:**
- All relevant skill types included (Programming languages, Foreign language, Lab skills)?
- All relevant skills listed within each skill type?

**Activities/Honors/Leadership Checklist:**
- Relevant activities, honors, and/or leadership experiences listed?
"""

# 10-Step Resume Checklist
ten_step_resume_checklist = """
**10-Step Resume Checklist:**
1.  **Structure:** Organized information concisely (Entry-level: Name, Contact, Summary, Education, Certs, Experience, Skills; Experienced: Name, Contact, Summary, Experience, Skills, Education, Certs).
2.  **Name and Contact Information:** First/last name in larger font, latest contact info (phone, email, city/province/territory), professional title (if applicable).
3.  **Professional Summary:** 3-5 sentences, highlights qualifications, experiences, skills, accomplishments.
4.  **Work Experience:** Listed in reverse chronological order, job title, employment dates, company name, city/province/territory for each. Past tense for past roles, present for current. Significant accomplishments (not just responsibilities), action words, quantified achievements. Keywords from job description included. 3-5 bullets using STAR method.
5.  **Skills:** Relevant hard skills (technical/professional) and soft skills (people/personal) included. Checked against job description requirements.
6.  **Education:** Degree, major, graduation date, school name. Reverse chronology. Anticipated graduation date if in school. (Graduation date optional if 5+ years ago).
7.  **Certifications/Other Training:** Optional section for professional experiences, technical expertise, certifications, awards, specific training.
8.  **Formatting:** Regular fonts (Times New Roman, Calibri, Arial), consistent font size (larger for name). Italics sparingly. Bold section headings. Saved as PDF.
9.  **Proofreading:** Thoroughly checked for spelling, grammar, typos. Verb phrases acceptable, no complete sentences. Acronyms spelled out, no pronouns. Peer review considered.
10. **Overall Style and Appearance:** Visually appealing, balance of text/white space. Margin at least 1/2" on all sides. Resume template considered.
"""

# New Comprehensive Resume Best Practices Prompt
lsm_resume_best_practices = """
**LSM Resume Best Practices Guide:**

**1. Do I need a cover letter?**
* **Not always necessary:** If a job posting doesn't specifically request one, you can often omit it.
* **When to include one:** Always if requested. Consider if you want to add context, applying for a competitive position, or have a personal connection.
* **Writing a Strong Cover Letter:** Concise, focused. Introduce yourself, summarize background (relevant experience/skills), showcase your value (why you're ideal candidate, unique qualifications).
* **Additional Tips:** Read job posting carefully. For email applications, email body is a brief introduction; attach formatted cover letter as separate document.

**2. Summary Section**
* Consider using a **resume summary or brand statement** (60-100 words) instead of an objective.
* **Why a Resume Summary?** Not always required if a cover letter is included. Valuable for quickly grabbing attention and highlighting qualifications, especially if no cover letter.
* **Writing an Effective Summary:** Focus on:
    * **Who you are and what you do:** Professional identity, expertise, alignment with job.
    * **Your achievements:** Highlight impressive, relevant accomplishments. Quantify results (e.g., increased sales by X%), awards, problems solved, challenges overcome, exceeding expectations.
    * **Goal:** Demonstrate skills, experience, and a proven track record of delivering results.

**3. Work Experience**
* **Purpose:** Detail career history, demonstrate relevant skills, experience, background. Typically longest section.
* **How to describe experience:**
    * Use **bullets** for skim value.
    * **Present roles:** Simple present tense (e.g., practices, manages).
    * **Past activities:** Simple past tense (e.g., achieved, managed).
    * **Include context:** Company type, products/services, customers, industry, size.
    * **Describe impact:** How your work supported the company/bottom line (products supported, customers supported, business unit worked for).
    * **Main functions:** Focus on things that occupy most time or are relevant to next job.
    * **Focus on impact:** Quantifiable achievements like landing clients, cost savings, automation, sales targets, awards, successful projects.
* **Bullet Point Structuring:**
    * **PAR (Problem-Action-Result):** Ideal for accomplishments, technical skills, problem-solving. Example: "Reduced customer support response time by 20% by implementing a new ticketing system..."
    * **STAR (Situation-Task-Action-Result):** Detailed narrative, suitable for behavioral questions. Example: "Managed inventory levels and ensured product availability...achieved a 15% reduction in inventory costs."
    * **Result-First:** Leads with quantifiable outcome for strong impact. Example: "Increased sales revenue by 18% through targeted marketing campaigns..."
    * **Action Verb + Skill + Result:** Concise, common for technical resumes. Example: "Developed a user-friendly website using HTML, CSS, and JavaScript, resulting in a 30% increase in website traffic."
* **Order:** Most recent experience first, reverse chronological order. Most recent role should have most bullet points.
* **Relevance:** Feature most relevant bullet points first for each application, aligning with job description.
* **DON'T DO THIS:** Generic statements, repetitive terms, overly fancy/flowery language.

**4. Gap Years or Employment Gaps**
* Be **transparent** while protecting privacy. Recruiters understand life happens.
* **How to show:** If personal leave, include "Personal leave of absence" with dates at top of experience section.

**5. Multiple Positions at One Company**
* **Similar scope:** Combine into a single title (e.g., "IT Specialist").
* **Overlap, not exact:** Choose most relevant title, add a promotion bullet (e.g., "Promoted internally from...").
* **Entirely exclusive:** List as separate positions.
* **Key:** Truthful simplicity. Never list a title/responsibilities you didn't hold.

**6. Multiple Non-Consecutive Terms or Discontinuous Work**
* **Longer work time than off-time:** Use single time range.
* **Short, distinct periods (e.g., summer jobs):** Use "Summers 2010/11/12/13".
* **Clarity over accuracy (without stretching truth).**

**7. Skills and Keywords**
* **Keywords:** Words/phrases from job posting, important skills. Including them increases visibility in recruiter searches.
* **Technical Skills:** Specific knowledge/expertise (tools, software, hardware, processes). Crucial for many roles.
    * **Hard Technical Skills:** Hands-on knowledge (e.g., Programming languages: Java, Python; Software applications: Microsoft Office Suite; OS: Windows; Data analysis: SQL; Cloud: AWS).
    * **Soft Technical Skills:** Using technical knowledge to solve problems, analyze data, communicate (e.g., Troubleshooting, Data analysis, Project management, Technical writing, Presentation skills).
    * **Incorporating using PAR:** Problem (or Project) + Action (technical skills used) + Result (quantified outcome).
    * **Soft Interpersonal Skills:** Terms like "Problem-solving," "Outgoing communicator." Overused, lack value. Best to leave off resume. Implied through well-written descriptions/accomplishments. Demonstrate (e.g., "Presented quarterly performance updates..." instead of "excellent communication skills").

**8. Skill Levels**
* **Do not provide self-evaluation** unless a conventional standard exists (e.g., certifications with levels). Assumption is you have working knowledge.

**9. Education**
* **Placement:** If very relevant or recent graduate, feature on first page above experience. Otherwise, below experience.
* **Coursework:** Selective; highlight 3-4 courses directly related to job.
* **Academic Projects:** Focus on outcomes and real-world applications (e.g., "Developed comprehensive growth strategy...contributed to a 20% projected revenue increase").
* **GPA:** Optional unless over 3.5 or specifically requested. Not applicable in all countries.
* **Honors:** Mention academic honors, dean's list, scholarships briefly.

**10. Interests/Hobbies**
* **Generally not needed.** More pertinent things to include. Avoid generic lists. If included, make it specific (e.g., "reading WWII history").

**11. References**
* **Do not include** references or "references available upon request" on resume. Bring a separate reference sheet to interviews.

**12. Resume Length**
* **Goal:** Pique interest, establish qualifications, create curiosity for an interview.
* **Most people:** One or two pages sufficient.
* **Exceptions:**
    * IT, cybersecurity, software engineering, government: 2-3 pages.
    * Senior executives: 3 pages.
    * Academia (CV): 5-12 pages.

**13. Projects & Portfolio (For Limited Work Experience)**
* **Purpose:** Proof of capability; demonstrate knowledge application, problem-solving, results.
* **Description Formula:** What you built + How you built it + The impact/outcome.
* **Group Projects:** Show your involvement while acknowledging teamwork (e.g., "Led 3-person team...").
* **Documentation:** Include links to GitHub, live sites, project docs, research posters, case studies.
* **Selection:** Prioritize projects aligned with target industry/role, relevant skills, measurable outcomes, collaborative work, self-directed projects.
* **Technical Roles:** Include technologies, languages, scale, complexity, challenges, performance improvements.
* **Quality over Quantity.**

**14. Experience Alternatives (When Traditional Work Experience is Limited)**
* Present activities to highlight **transferable skills and measurable impact**.
* **Examples:**
    * **Leadership in Student Organizations:** (e.g., "Led 5-person team to increase club membership by 40%...").
    * **Volunteer Work:** (e.g., "Developed and implemented new donor tracking system...improving follow-up response rate by 25%").
    * **Academic Research Experience (Unpaid):** (e.g., "Assisted professor with data collection and analysis...").
    * **Part-time Work Reframed:** (e.g., "Maintained 98% customer satisfaction while processing 200+ transactions...").
* Focus on responsibilities demonstrating initiative, reliability, measurable results.

**15. How you present your content matters**
* Turn basic task descriptions into **achievement statements** showcasing impact.
* **Start bullets with strong action verbs:** "Initiated," "Led," "Created," "Managed," "Delivered."
* **Quantify wherever possible:** (e.g., "Increased Instagram engagement by 25%," "Reduced processing time by 15%").
* **If no numbers, focus on scope/scale:** (e.g., "Coordinated logistics for three campus-wide events," "Maintained detailed documentation used by entire 20-person organization").
* Be specific and detailed to show value, even without formal work experience.
"""

st.sidebar.subheader("LSM Resume Checklist Options")
use_general_checklist = st.sidebar.checkbox("Include General Resume Checklist in Prompt")
use_10_step_checklist = st.sidebar.checkbox("Include 10-Step Resume Checklist in Prompt")
use_lsm_best_practices = st.sidebar.checkbox("Include LSM Resume Best Practices in Prompt", value=True)

# --- API Output Option ---
st.sidebar.subheader("LSM API Output Options")
api_output_option = st.sidebar.radio(
    "Select API Output Format:",
    ('Streamlit App Display', 'OpenAPI JSON (for integration)')
)

# --- Button Layout ---
col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    submit1 = st.button("Tell Me About the Resume (LSM)")

with col2:
    submit2 = st.button("Get Keywords (LSM)")

with col3:
    submit3 = st.button("Percentage Match (LSM)")

# --- Prompt Definitions ---
input_prompt1_base = """
You are an experienced Technical Human Resource Manager. Your task is to review the provided resume against the job description.
Please share your professional evaluation on whether the candidate's profile aligns with the role.
Highlight the strengths and weaknesses of the applicant in relation to the specified job requirements.
"""

input_prompt2_base = """
As an expert ATS (Applicant Tracking System) scanner with an in-depth understanding of AI and ATS functionality,
your task is to evaluate a resume against a provided job description. Please identify the specific skills and keywords
necessary to maximize the impact of the resume and provide response in json format as {Technical Skills:[], Analytical Skills:[], Soft Skills:[]}.
Note: Please do not make up the answer; only answer from the job description provided.
"""

input_prompt3_base = """
You are a skilled ATS (Applicant Tracking System) scanner with a deep understanding of data science and ATS functionality.
Your task is to evaluate the resume against the provided job description. Give me the percentage of match if the resume matches
the job description. First, the output should come as percentage and then keywords missing and last final thoughts.
"""

# --- Dynamic Prompt Construction and Execution ---
def get_full_prompt_and_params(base_prompt):
    additional_params = {}
    if use_general_checklist:
        additional_params["General Resume Checklist"] = general_resume_checklist
    if use_10_step_checklist:
        additional_params["10-Step Resume Checklist"] = ten_step_resume_checklist
    if use_lsm_best_practices:
        additional_params["LSM Resume Best Practices Guide"] = lsm_resume_best_practices
    return base_prompt, additional_params

if submit1:
    if uploaded_file is None:
        st.write("Please upload the resume.")
    elif ai_model is None:
        st.error(f"Selected AI model ({ai_model_choice}) not initialized. Please ensure the correct API key is set in `secrets.toml`.")
    else:
        try:
            pdf_content = input_pdf_setup(uploaded_file)
            base_prompt, additional_params = get_full_prompt_and_params(input_prompt1_base)
            response = get_ai_response(ai_model_choice, ai_model, base_prompt, pdf_content, input_text, additional_params)

            if api_output_option == 'Streamlit App Display':
                st.subheader("The Response is (LSM)")
                st.write(response)
            else: # OpenAPI JSON
                st.subheader("OpenAPI JSON Output (LSM)")
                st.json({"evaluation": response})
        except (FileNotFoundError, ValueError) as e:
            st.error(f"Error: {e}")

elif submit2:
    if uploaded_file is None:
        st.write("Please upload the resume.")
    elif ai_model is None:
        st.error(f"Selected AI model ({ai_model_choice}) not initialized. Please ensure the correct API key is set in `secrets.toml`.")
    else:
        try:
            pdf_content = input_pdf_setup(uploaded_file)
            base_prompt, additional_params = get_full_prompt_and_params(input_prompt2_base)
            response = get_ai_response_keywords(ai_model_choice, ai_model, base_prompt, pdf_content, input_text, additional_params)

            if api_output_option == 'Streamlit App Display':
                st.subheader("Skills are (LSM):")
                if response:
                    st.write(f"**Technical Skills:** {', '.join(response.get('Technical Skills', []))}.")
                    st.write(f"**Analytical Skills:** {', '.join(response.get('Analytical Skills', []))}.")
                    st.write(f"**Soft Skills:** {', '.join(response.get('Soft Skills', []))}.")
                else:
                    st.write("No skills found or an error occurred during processing.")
            else: # OpenAPI JSON
                st.subheader("OpenAPI JSON Output (LSM)")
                st.json({"keywords": response})
        except (FileNotFoundError, ValueError) as e:
            st.error(f"Error: {e}")

elif submit3:
    if uploaded_file is None:
        st.write("Please upload the resume.")
    elif ai_model is None:
        st.error(f"Selected AI model ({ai_model_choice}) not initialized. Please ensure the correct API key is set in `secrets.toml`.")
    else:
        try:
            pdf_content = input_pdf_setup(uploaded_file)
            base_prompt, additional_params = get_full_prompt_and_params(input_prompt3_base)
            response = get_ai_response(ai_model_choice, ai_model, base_prompt, pdf_content, input_text, additional_params)

            if api_output_option == 'Streamlit App Display':
                st.subheader("The Response is (LSM)")
                st.write(response)
            else: # OpenAPI JSON
                st.subheader("OpenAPI JSON Output (LSM)")
                st.json({"match_analysis": response})
        except (FileNotFoundError, ValueError) as e:
            st.error(f"Error: {e}")
