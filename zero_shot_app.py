from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def classify_text(text_to_classify, labels_string):
    if not labels_string:
        return "Error: Please provide candidate labels."
        
    candidate_labels = [label.strip() for label in labels_string.split(',')]

    result = classifier(text_to_classify, candidate_labels)

    output = []
    for label, score in zip(result['labels'], result['scores']):
        output.append(f"{label}: {score:.2f}") 
        
    return "\n".join(output)


import gradio as gr 
iface = gr.Interface(
    fn=classify_text,
    
    inputs=[
        gr.Textbox(
            lines=5, 
            label="Enter Text for Classification", 
            placeholder="Example: My account is locked after the update."
        ),
        gr.Textbox(
            lines=3,
            label="Candidate Labels (Comma Separated)",
            value="Technical Issue, Billing Query, Feature Request, General Feedback" 
        )
    ],
    
    outputs=gr.Text(
        label="Classification Results (Label: Probability)",
        lines=10 
    ),
    
    title="Interactive Zero-Shot Intent Recognition Demo",
    description="Modify the labels instantly to see how the model adapts its classification."
)

iface.launch()