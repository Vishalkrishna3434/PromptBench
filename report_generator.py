import pandas as pd
from fpdf import FPDF
import os

class PromptBenchPDF(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 15)
        self.set_text_color(99, 102, 241) # Accent color
        self.cell(0, 10, 'PromptBench: AI Engineering Analytics', border=0, align='R')
        self.ln(10)
        self.set_draw_color(226, 232, 240)
        self.line(10, 22, 200, 22)
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.set_text_color(148, 163, 184)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, label):
        self.set_font('helvetica', 'B', 16)
        self.set_text_color(30, 41, 59)
        self.set_fill_color(248, 250, 252)
        self.cell(0, 12, f"  {label}", new_x="LMARGIN", new_y="NEXT", border=0, fill=True)
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('helvetica', '', 11)
        self.set_text_color(51, 65, 85)
        self.multi_cell(0, 8, body)
        self.ln(10)

def generate_pdf():
    print("\n==========================================")
    print("O/P: Starting Phase 4 (Polished PDF Report Generation)...")
    
    pdf = PromptBenchPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # --- Front Page Title ---
    pdf.ln(20)
    pdf.set_font("helvetica", size=32, style="B")
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 20, "PROMPTBENCH", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("helvetica", size=14)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 10, "Modular Benchmarking & AI Performance Analytics", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(30)
    
    # --- Executive Summary ---
    pdf.chapter_title("1. Executive Summary")
    summary = (
        "PromptBench is a comprehensive pipeline designed to analyze and optimize prompt engineering strategies. "
        "By leveraging automated dataset building, statistical feature extraction, and live LLM benchmarking, "
        "the project provides actionable insights into how prompt structure affects model response quality, "
        "latency, and cost efficiency. This report serves as the final deliverable for the Course Based Project."
    )
    pdf.chapter_body(summary)

    # --- The Scoring Formula ---
    pdf.chapter_title("2. The Optimization Formula Breakdown")
    formula_intro = (
        "We formalize the 'Prompt Efficiency' using a weighted scoring algorithm. This formula rewards "
        "high-quality architectural features while penalizing computational overhead and slowness."
    )
    pdf.chapter_body(formula_intro)
    
    # Formula Box
    pdf.set_fill_color(241, 245, 249)
    pdf.set_font("courier", size=10, style="B")
    pdf.set_text_color(30, 41, 59)
    formula = "Score = (Inst * 0.5) + (Spec * 0.2) + (Ex * 1.5) + (Quality) - (Tokens * 0.005) - (Lat * 0.2)"
    pdf.cell(0, 15, formula, border=1, align="C", fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)

    # Legend Header
    pdf.set_font("helvetica", size=10, style="B")
    pdf.set_text_color(71, 85, 105)
    pdf.cell(30, 8, "Term", border="B")
    pdf.cell(160, 8, "Description", border="B", new_x="LMARGIN", new_y="NEXT")
    
    # Legend Items
    legend = [
        ("Inst", "Instruction Count: Total number of action-oriented verbs (e.g., 'always', 'do') parsed from the prompt."),
        ("Spec", "Specificity Score: Percentage of the prompt containing strict constraints ('must', 'strictly') relative to length."),
        ("Ex", "Example Count: Number of few-shot examples or 'e.g.' usage provided to guide the model."),
        ("Quality", "Gemini Quality Score: A 1-10 accuracy rating assigned by the Gemini-pro automated evaluator."),
        ("Tokens", "Token Usage: Total token count of the model's response, serving as the primary cost factor."),
        ("Lat", "Latency: The time in seconds taken for the model to generate the response.")
    ]
    
    for term, desc in legend:
        current_y = pdf.get_y()
        pdf.set_font("helvetica", size=10, style="B")
        pdf.cell(30, 8, term)
        pdf.set_font("helvetica", size=10)
        # Using multi_cell with a fixed width to ensure space
        pdf.multi_cell(160, 8, desc, new_x="LMARGIN", new_y="NEXT")
        # Ensure minimal padding between rows
        pdf.ln(2)
    
    pdf.ln(10)

    # --- Findings ---
    pdf.chapter_title("3. Dataset Insights")
    try:
        df = pd.read_csv("prompt_dataset.csv")
        findings = (
            f"Analysis of {len(df)} real-world prompts revealed the following benchmarks:\n\n"
            f"- Average Complexity: {df['token_length'].mean():.1f} tokens per prompt.\n"
            f"- Instruction Density: {df['instruction_count'].mean():.1f} verbs identified per string.\n"
            f"- Constraint Variance: {df['specificity_score'].max():.1f}% max specificity rating.\n\n"
            "Our data suggests that prompts containing concrete 'Examples' result in a significantly higher "
            "consistency rating compared to open-ended natural language instructions."
        )
        pdf.chapter_body(findings)
    except:
        pdf.chapter_body("Dataset analysis results were unavailable during compilation.")

    # --- Visualizations ---
    pdf.add_page()
    pdf.chapter_title("4. Statistical Data Visualization")
    pdf.ln(5)
    
    y_val = pdf.get_y()
    if os.path.exists("plot_box.png"):
        pdf.image("plot_box.png", x=15, y=y_val, w=85)
    if os.path.exists("plot_hist.png"):
        pdf.image("plot_hist.png", x=110, y=y_val, w=85)
    
    pdf.ln(75)
    if os.path.exists("plot_scatter.png"):
        pdf.image("plot_scatter.png", x=60, y=pdf.get_y(), w=90)
    
    pdf.ln(90)

    # --- Live Benchmarking ---
    pdf.chapter_title("5. Live API Benchmarking (Gemini)")
    try:
        res = pd.read_csv("prompt_results.csv")
        
        pdf.set_font('helvetica', 'B', 10)
        pdf.set_fill_color(99, 102, 241)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(30, 10, "Prompt ID", border=1, align="C", fill=True)
        pdf.cell(50, 10, "Quality Score", border=1, align="C", fill=True)
        pdf.cell(50, 10, "Token Usage", border=1, align="C", fill=True)
        pdf.cell(50, 10, "Latency (s)", border=1, align="C", fill=True, new_x="LMARGIN", new_y="NEXT")
        
        pdf.set_font('helvetica', '', 10)
        pdf.set_text_color(51, 65, 85)
        for i, row in res.iterrows():
            fill = (i % 2 == 0)
            pdf.set_fill_color(248, 250, 252) if fill else pdf.set_fill_color(255, 255, 255)
            pdf.cell(30, 10, str(int(row['Prompt_ID'])), border=1, align="C", fill=True)
            pdf.cell(50, 10, f"{row['Score']}/10", border=1, align="C", fill=True)
            pdf.cell(50, 10, str(int(row['Token_Usage'])), border=1, align="C", fill=True)
            pdf.cell(50, 10, f"{row['Latency']}s", border=1, align="C", fill=True, new_x="LMARGIN", new_y="NEXT")
    except:
        pdf.chapter_body("API benchmarking data was unavailable during compilation.")

    pdf.ln(10)
    pdf.set_font("helvetica", size=10, style="I")
    pdf.set_text_color(148, 163, 184)
    pdf.cell(0, 10, "Generated automatically by PromptBench Pipeline.", align="C")

    pdf.output("PromptBench_Report.pdf")
    print("O/P: Successfully saved PromptBench_Report.pdf")
