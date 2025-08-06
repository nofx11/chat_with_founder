from dotenv import load_dotenv
from agents import Agent, Runner, trace, gen_trace_id
from pypdf import PdfReader
import gradio as gr
from utils import _record_user_details, trim_history


load_dotenv(override=True)
MAX_TOKENS = 1000
name = "Daniel Polonski"
reader = PdfReader("me/linkedin.pdf")
linkedin = ""
for page in reader.pages:
    text = page.extract_text()
    if text:
        linkedin += text
with open("me/summary.txt", "r", encoding="utf-8") as f:
    summary = f.read()

INSTRUCTIONS = f"""
You are acting as {name}. You are answering questions on {name}'s website,
particularly questions related to {name}'s career, background, skills and experience.
Your responsibility is to represent {name} for interactions on the website as faithfully as possible.
You are given a summary of {name}'s background and LinkedIn profile which you can use to answer questions.
Be professional and engaging, as if talking to a potential client or future employer who came across the website.
If the user is engaging in discussion, try to steer them towards getting in touch via email; ask for their email and record it using your record_user_details tool.
"""

INSTRUCTIONS += f"\n\n## Summary:\n{summary}\n\n## LinkedIn Profile:\n{linkedin}\n\n"
INSTRUCTIONS += f"With this context, please chat with the user, always staying in character as {name}."


chat_agent = Agent(
    name="chat_agent",
    instructions=INSTRUCTIONS,
    tools=[_record_user_details],
    model="gpt-4o-mini"
)

async def chat(message, history):
    trace_id = gen_trace_id()
    with trace("chat trace", trace_id=trace_id):
        print(f"View trace: https://platform.openai.com/traces/trace?trace_id={trace_id}")
        processed_history = []
        for msg in history:
            if isinstance(msg, dict):
                processed_history.append({"role": msg['role'], "content": msg['content']})
            elif isinstance(msg, (list, tuple)) and len(msg) == 2:
                # Gradio starszy: (user_msg, assistant_msg)
                processed_history.append({"role": "user", "content": msg[0]})
                processed_history.append({"role": "assistant", "content": msg[1]})
            else:
                print(f"Nieznany format historii: {msg}")

        processed_history.append({"role": "user", "content": message})
        processed_history = trim_history(processed_history, MAX_TOKENS)
        #print(f"Processed history: {processed_history}")
        result = await Runner.run(chat_agent, processed_history)
    return result.final_output


force_dark_mode = """
function refresh() {
    const url = new URL(window.location);
    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.href = url.href;
    }
}
"""

# UI w Gradio
if __name__ == "__main__":
    interface = gr.ChatInterface(
        fn=chat,
        title="Chat with Daniel Polonski",
        description="Ask about Daniel's experience, skills, or career background.",
        # theme=gr.themes.Default(primary_hue="sky"),
        js=force_dark_mode,   #theme: alternatywnie: "gradio.dark" , soft
        examples=[
            "What do you do?",
            "What is your experience in AI/ML?",
            "Which companies have you worked for?",
            "How can I contact you?"
        ]
    )

    interface.launch()
    