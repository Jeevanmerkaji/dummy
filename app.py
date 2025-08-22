import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import pandas as pd
from src.helper import loader_pdf_file, filter_to_minimal_docs, split_text, download_hugging_face_embeddings
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from src.prompt import *
import os 
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate



# Initialize the Dash app
app = dash.Dash(__name__)


# Custom CSS styling
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("AI Medical Diagnosis Assistant", 
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '10px'}),
        html.P("Professional AI-Powered Diagnostic Support System", 
               style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '16px'})
    ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'marginBottom': '30px'}),
    
    # Main container
    html.Div([
        # Left panel - Patient Information Input
        html.Div([
            html.H3("Patient Information", style={'color': '#2c3e50', 'marginBottom': '20px'}),
            
            # Patient Demographics
            html.Div([
                html.Label("Patient Name:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                dcc.Input(
                    id='patient-name',
                    type='text',
                    placeholder='Enter patient name...',
                    style={'width': '100%', 'padding': '10px', 'marginBottom': '15px', 
                           'border': '1px solid #ddd', 'borderRadius': '5px'}
                ),
                
                html.Div([
                    html.Div([
                        html.Label("Age:", style={'fontWeight': 'bold'}),
                        dcc.Input(
                            id='patient-age',
                            type='number',
                            placeholder='Age',
                            min=0, max=120,
                            style={'width': '100%', 'padding': '10px', 'border': '1px solid #ddd', 'borderRadius': '5px'}
                        )
                    ], style={'width': '48%', 'display': 'inline-block'}),
                    
                    html.Div([
                        html.Label("Gender:", style={'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='patient-gender',
                            options=[
                                {'label': 'Male', 'value': 'male'},
                                {'label': 'Female', 'value': 'female'},
                                {'label': 'Other', 'value': 'other'}
                            ],
                            placeholder="Select gender",
                            style={'border': '1px solid #ddd', 'borderRadius': '5px'}
                        )
                    ], style={'width': '48%', 'float': 'right'})
                ], style={'marginBottom': '15px'}),
            ]),
            
            # Chief Complaint
            html.Label("Chief Complaint:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
            dcc.Textarea(
                id='chief-complaint',
                placeholder='Describe the main symptoms or concerns...',
                style={'width': '100%', 'height': '80px', 'padding': '10px', 'marginBottom': '15px',
                       'border': '1px solid #ddd', 'borderRadius': '5px', 'resize': 'vertical'}
            ),
            
            # Medical History
            html.Label("Medical History:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
            dcc.Textarea(
                id='medical-history',
                placeholder='Previous conditions, surgeries, medications...',
                style={'width': '100%', 'height': '80px', 'padding': '10px', 'marginBottom': '15px',
                       'border': '1px solid #ddd', 'borderRadius': '5px', 'resize': 'vertical'}
            ),
            
            # Current Symptoms
            html.Label("Current Symptoms:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
            dcc.Textarea(
                id='current-symptoms',
                placeholder='Detailed description of current symptoms...',
                style={'width': '100%', 'height': '100px', 'padding': '10px', 'marginBottom': '15px',
                       'border': '1px solid #ddd', 'borderRadius': '5px', 'resize': 'vertical'}
            ),
            
            # Vital Signs
            html.Label("Vital Signs:", style={'fontWeight': 'bold', 'marginBottom': '10px'}),
            html.Div([
                html.Div([
                    html.Label("Blood Pressure:", style={'fontSize': '14px'}),
                    dcc.Input(
                        id='blood-pressure',
                        type='text',
                        placeholder='120/80',
                        style={'width': '100%', 'padding': '8px', 'border': '1px solid #ddd', 'borderRadius': '5px'}
                    )
                ], style={'width': '48%', 'display': 'inline-block', 'marginBottom': '10px'}),
                
                html.Div([
                    html.Label("Heart Rate:", style={'fontSize': '14px'}),
                    dcc.Input(
                        id='heart-rate',
                        type='number',
                        placeholder='72',
                        style={'width': '100%', 'padding': '8px', 'border': '1px solid #ddd', 'borderRadius': '5px'}
                    )
                ], style={'width': '48%', 'float': 'right', 'marginBottom': '10px'}),
                
                html.Div([
                    html.Label("Temperature (°F):", style={'fontSize': '14px'}),
                    dcc.Input(
                        id='temperature',
                        type='number',
                        placeholder='98.6',
                        step=0.1,
                        style={'width': '100%', 'padding': '8px', 'border': '1px solid #ddd', 'borderRadius': '5px'}
                    )
                ], style={'width': '48%', 'display': 'inline-block'}),
                
                html.Div([
                    html.Label("Respiratory Rate:", style={'fontSize': '14px'}),
                    dcc.Input(
                        id='respiratory-rate',
                        type='number',
                        placeholder='16',
                        style={'width': '100%', 'padding': '8px', 'border': '1px solid #ddd', 'borderRadius': '5px'}
                    )
                ], style={'width': '48%', 'float': 'right'})
            ], style={'marginBottom': '20px'}),
            
            # Submit Button
            html.Button(
                'Get AI Diagnosis',
                id='submit-button',
                n_clicks=0,
                style={
                    'width': '100%',
                    'padding': '15px',
                    'backgroundColor': '#3498db',
                    'color': 'white',
                    'border': 'none',
                    'borderRadius': '5px',
                    'fontSize': '16px',
                    'fontWeight': 'bold',
                    'cursor': 'pointer',
                    'marginBottom': '20px'
                }
            ),
            
            # Clear Button
            html.Button(
                'Clear All Fields',
                id='clear-button',
                n_clicks=0,
                style={
                    'width': '100%',
                    'padding': '10px',
                    'backgroundColor': '#95a5a6',
                    'color': 'white',
                    'border': 'none',
                    'borderRadius': '5px',
                    'fontSize': '14px',
                    'cursor': 'pointer'
                }
            )
            
        ], style={
                'flex': '0 0 45%',
                'padding': '20px',
                'backgroundColor': '#f8f9fa',
                'borderRadius': '10px',
                'marginRight': '2%'
        }),
        
        # Right panel - AI Response
        html.Div([
            html.H3("AI Diagnostic Analysis", style={'color': '#2c3e50', 'marginBottom': '20px'}),
            
            # Loading indicator
            dcc.Loading(
                id="loading",
                type="circle",
                children=[
                    html.Div(id='diagnosis-output', style={
                        'minHeight': '400px',
                        'padding': '20px',
                        'backgroundColor': 'white',
                        'border': '1px solid #ddd',
                        'borderRadius': '5px',
                        'fontSize': '14px',
                        'lineHeight': '1.6'
                    })
                ]
            ),
            
            # Confidence Score
            html.Div(id='confidence-score', style={'marginTop': '20px'}),
            
            # Recommendations
            html.Div([
                html.H4("Recommended Actions:", style={'color': '#2c3e50', 'marginTop': '30px'}),
                html.Div(id='recommendations', style={
                    'padding': '15px',
                    'backgroundColor': '#e8f5e8',
                    'border': '1px solid #c8e6c9',
                    'borderRadius': '5px',
                    'marginTop': '10px'
                })
            ], id='recommendations-section', style={'display': 'none'})
            
        ], style={
                'flex': '0 0 53%',
                'padding': '20px',
                'backgroundColor': '#f8f9fa',
                'borderRadius': '10px'
        })
        
    ], style={'margin': '0 20px',
            'display': 'flex',
            'flexDirection': 'row',
            'justifyContent': 'space-between',
            'gap': '2%'
        }),
    
    # Footer
    html.Div([
        html.P("⚠️ This AI system is for diagnostic assistance only. Always use clinical judgment and follow proper medical protocols.",
               style={'textAlign': 'center', 'color': '#e74c3c', 'fontSize': '12px', 'fontStyle': 'italic'})
    ], style={'marginTop': '30px', 'padding': '20px', 'backgroundColor': '#ecf0f1'})
    
], style={'fontFamily': 'Arial, sans-serif', 'margin': '0', 'backgroundColor': '#ffffff'})

# Callback for getting AI diagnosis
@app.callback(
    [Output('diagnosis-output', 'children'),
     Output('confidence-score', 'children'),
     Output('recommendations', 'children'),
     Output('recommendations-section', 'style')],
    [Input('submit-button', 'n_clicks')],
    [State('patient-name', 'value'),
     State('patient-age', 'value'),
     State('patient-gender', 'value'),
     State('chief-complaint', 'value'),
     State('medical-history', 'value'),
     State('current-symptoms', 'value'),
     State('blood-pressure', 'value'),
     State('heart-rate', 'value'),
     State('temperature', 'value'),
     State('respiratory-rate', 'value')]
)
def get_diagnosis(n_clicks, name, age, gender, complaint, history, symptoms, bp, hr, temp, rr):
    if n_clicks == 0:
        return [
            html.Div([
                html.H4("Welcome to AI Medical Diagnosis Assistant", style={'color': '#3498db'}),
                html.P("Please fill out the patient information form and click 'Get AI Diagnosis' to receive diagnostic suggestions."),
                html.Hr(),
                html.H5("How to use:"),
                html.Ul([
                    html.Li("Enter patient demographics (name, age, gender)"),
                    html.Li("Describe the chief complaint and symptoms"),
                    html.Li("Include relevant medical history"),
                    html.Li("Add vital signs if available"),
                    html.Li("Click 'Get AI Diagnosis' for analysis")
                ]),
                html.P("The AI will provide potential diagnoses, confidence scores, and recommended next steps.",
                       style={'marginTop': '20px', 'fontStyle': 'italic'})
            ])
        ], "", "", {'display': 'none'}
    
    # Validate required fields
    if not complaint and not symptoms:
        return [
            html.Div([
                html.H4("⚠️ Missing Information", style={'color': '#e74c3c'}),
                html.P("Please provide at least the chief complaint or current symptoms to proceed with diagnosis.")
            ])
        ], "", "", {'display': 'none'}
    
    # Prepare patient data for LLM
    patient_data = {
        'name': name or 'Not provided',
        'age': age or 'Not provided',
        'gender': gender or 'Not provided',
        'chief_complaint': complaint or 'Not provided',
        'medical_history': history or 'Not provided',
        'current_symptoms': symptoms or 'Not provided',
        'vitals': {
            'blood_pressure': bp or 'Not provided',
            'heart_rate': hr or 'Not provided',
            'temperature': temp or 'Not provided',
            'respiratory_rate': rr or 'Not provided'
        }
    }
    
    # TODO: Replace this section with your actual LLM model call
    # Here's where you would integrate your LLM model
    # diagnosis_response = your_llm_model.predict(patient_data)
    
    diagnosis_response = call_llm_model(patient_data)


    # Simulated AI response for demonstration
    simulated_diagnosis = f"""
    **Patient Summary:** {patient_data['name']}, {patient_data['age']} years old, {patient_data['gender']}
    
    **Chief Complaint:** {patient_data['chief_complaint']}
    
    **Analysis:**
    Based on the provided symptoms and patient information, the AI has identified several potential diagnoses:
    
    **Primary Differential Diagnoses:**
    1. **Viral Upper Respiratory Infection** (Confidence: 75%)
       - Symptoms align with common viral presentation
       - Age and symptom duration support this diagnosis
    
    2. **Bacterial Sinusitis** (Confidence: 60%)
       - Consider if symptoms persist beyond 7-10 days
       - May require antibiotic treatment
    
    3. **Allergic Rhinitis** (Confidence: 45%)
       - Seasonal patterns should be considered
       - Environmental triggers may be relevant
    
    **Clinical Reasoning:**
    The combination of symptoms, patient demographics, and vital signs suggests a respiratory etiology. The presentation is most consistent with a viral process, though bacterial superinfection should be monitored.
    
    **Risk Stratification:** Low to Moderate Risk
    """
    
    confidence_display = html.Div([
        html.H5("Diagnostic Confidence", style={'color': '#2c3e50'}),
        dcc.Graph(
            figure=go.Figure(data=[
                go.Bar(x=['Viral URI', 'Bacterial Sinusitis', 'Allergic Rhinitis'], 
                       y=[75, 60, 45],
                       marker_color=['#2ecc71', '#f39c12', '#e74c3c'])
            ]).update_layout(
                title="Confidence Scores (%)",
                yaxis=dict(range=[0, 100]),
                height=200,
                margin=dict(l=0, r=0, t=30, b=0)
            ),
            config={'displayModeBar': False}
        )
    ])
    
    recommendations_content = html.Div([
        html.Ul([
            html.Li("Order complete blood count (CBC) if bacterial infection suspected"),
            html.Li("Consider chest X-ray if respiratory symptoms worsen"),
            html.Li("Symptomatic treatment with rest, fluids, and over-the-counter medications"),
            html.Li("Follow-up in 7-10 days if symptoms persist or worsen"),
            html.Li("Patient education on warning signs requiring immediate medical attention")
        ])
    ])
    
    return [
        html.Div([
            dcc.Markdown(diagnosis_response, style={'whiteSpace': 'pre-line'})
        ]),
        confidence_display,
        recommendations_content,
        {'display': 'block'}
    ]

# Callback for clearing all fields
@app.callback(
    [Output('patient-name', 'value'),
     Output('patient-age', 'value'),
     Output('patient-gender', 'value'),
     Output('chief-complaint', 'value'),
     Output('medical-history', 'value'),
     Output('current-symptoms', 'value'),
     Output('blood-pressure', 'value'),
     Output('heart-rate', 'value'),
     Output('temperature', 'value'),
     Output('respiratory-rate', 'value')],
    [Input('clear-button', 'n_clicks')]
)
def clear_fields(n_clicks):
    if n_clicks > 0:
        return [None] * 10
    return dash.no_update

# Function to integrate your LLM model
def call_llm_model(patient_data):
    """
    Replace this function with your actual LLM model integration
    
    Args:
        patient_data (dict): Dictionary containing patient information
        
    Returns:
        dict: Diagnosis results from your LLM model
    """
    # Example structure for your LLM integration:
    # 
    # prompt = f"""
    # You are a medical AI assistant. Based on the following patient information,
    # provide differential diagnoses with confidence scores and recommendations.
    # 
    # Patient Data:
    # {patient_data}
    # 
    # Please provide:
    # 1. Top 3 differential diagnoses with confidence scores
    # 2. Clinical reasoning
    # 3. Recommended next steps
    # """
    # 
    # response = your_model.generate(prompt)
    # return response
    
    GROQ_API_KEY = os.environ["GROQ_API_KEY"] = "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"
    os.environ["GROQ_API_KEY"] =  GROQ_API_KEY

     # GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    embeddings = download_hugging_face_embeddings()
    extracted_data =  loader_pdf_file(data='data/')
    filter_data = filter_to_minimal_docs(extracted_data)
    text_chunks =  split_text(filter_data)

    texts = [doc.page_content for doc in text_chunks]
    metadatas = [doc.metadata for doc in text_chunks]  


    chat_model = ChatGroq(
    model_name="llama3-70b-8192"
    )

    prompt =  ChatPromptTemplate.from_messages(
    [
        ("system" , system_prompt),
        ("human","{input}")
    ]
    )
    # Create FAISS vector store
    vectorstore = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)
    retriever  = vectorstore.as_retriever(search_type = "similarity", search_kwargs ={"k":3})
    question_answer_chain =  create_stuff_documents_chain(chat_model, prompt)
    rag_chain =  create_retrieval_chain(retriever, question_answer_chain)
    print("Here I am trying to do somestuff")
    try:
        input_text = f"""
            Patient Name: {patient_data['name']}
            Age: {patient_data['age']}
            Gender: {patient_data['gender']}

            Chief Complaint:
            {patient_data['chief_complaint']}

            Medical History:
            {patient_data['medical_history']}

            Current Symptoms:
            {patient_data['current_symptoms']}

            Vital Signs:
            - Blood Pressure: {patient_data['vitals']['blood_pressure']}
            - Heart Rate: {patient_data['vitals']['heart_rate']}
            - Temperature: {patient_data['vitals']['temperature']}
            - Respiratory Rate: {patient_data['vitals']['respiratory_rate']}
            """
        
        response = rag_chain.invoke({"input": input_text})
        # print("Response : ", response['answer'])
        return str(response["answer"])
    except Exception as e:
        print("LLM Error:", e)
        return "⚠️ Error: The AI model failed to respond. Please check logs or try again later."
        

if __name__ == '__main__':
    app.run(debug=True,use_reloader = False)
