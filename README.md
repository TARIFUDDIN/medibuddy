# Create README.md with the project description
echo "# MediBot - Medical Assistant Chatbot

## Description

MediBot is an intelligent medical assistant chatbot designed to provide accurate and reliable medical information sourced from trusted medical texts and research papers. Built with modern AI technology, the application allows users to ask medical questions in natural language and receive informative responses backed by cited medical sources.

### Key Features

- **AI-Powered Medical Knowledge**: Utilizes advanced language models (Mistral-7B and FLAN-T5) to interpret medical questions and deliver accurate responses
- **Evidence-Based Information**: Retrieves information from a curated database of medical literature, ensuring answers are grounded in medical science
- **Source Citations**: Every response includes references to the source documents, providing transparency and allowing users to verify information
- **User-Friendly Interface**: Clean, intuitive Streamlit interface makes it easy for anyone to access medical information
- **Customizable AI Models**: Users can select between different AI models to power their experience
- **Privacy-Focused**: Processes queries locally without storing conversation data

### Use Cases

- Quick access to basic medical information and explanations
- Understanding medical terminology and concepts
- Learning about symptoms, conditions, treatments, and medications
- Educational tool for medical students and healthcare professionals
- Preliminary research tool for patients before consulting healthcare providers

### Important Note

MediBot is designed for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare providers for medical concerns. The application aims to make medical information more accessible while maintaining accuracy through its citation system." > README.md

# Add and commit README
git add README.md
git commit -m "Add README.md with project description"
git push origin main