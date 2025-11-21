"""
Google Gemini Integration Module
Handles all interactions with Gemini API for agricultural assistance
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai
from pathlib import Path

# Load environment variables
load_dotenv()


class GeminiHelper:
    """Helper class for Google Gemini API interactions"""
    
    def __init__(self, api_key=None):
        """
        Initialize Gemini Helper
        
        Args:
            api_key: Google Gemini API key (if None, loads from environment)
        """
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. Please set it in .env file or pass it as parameter."
            )
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        
        # Initialize model
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        
        # System prompt for agricultural context
        self.agricultural_system_prompt = """You are an expert agricultural advisor. Provide detailed, well-structured responses between 300-500 words.

MANDATORY FORMAT - Include ALL sections:

**Overview**
Write 3-4 sentences describing what this condition is and its impact on plants.

**Main Causes**
Explain the primary causes in detail (3-4 sentences). Include environmental factors, pathogens, or conditions that lead to this problem.

**Symptoms to Look For**
List 3-4 specific visual symptoms farmers should observe to confirm the diagnosis.

**Treatment Steps**
Provide detailed treatment with 4-5 specific steps:
1. Immediate action (what to do right now)
2. Chemical treatment (specific product names if possible)
3. Organic alternatives
4. Application method and frequency
5. Expected timeline for recovery

**Prevention Measures**
Provide 3-4 detailed prevention strategies:
- Cultural practices
- Soil management
- Water management
- Crop rotation or spacing

**Additional Care Tips**
Include 2-3 sentences about ongoing maintenance, nutrition, or monitoring.

IMPORTANT: Write at least 300 words. Be specific and practical. Use clear headings exactly as shown above."""
    
    def get_text_response(self, user_query):
        """
        Get response for a text-based agricultural query
        
        Args:
            user_query: User's question or description of symptoms
            
        Returns:
            Gemini's response as string
        """
        # Construct the full prompt
        full_prompt = f"""{self.agricultural_system_prompt}

Query: {user_query}

Provide a brief answer:"""
        
        try:
            response = self.model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def get_image_diagnosis_response(self, predicted_disease, confidence, user_query=None):
        """
        Get response for image-based diagnosis with CNN model prediction
        
        Args:
            predicted_disease: Disease name predicted by CNN model
            confidence: Confidence score of the prediction (0-1)
            user_query: Optional additional context from user
            
        Returns:
            Gemini's response as string
        """
        # Construct context-aware prompt
        confidence_percent = confidence * 100
        
        context = f"""I have analyzed a plant image using a trained CNN model for plant disease detection.

Model Prediction Results:
- Detected Condition: {predicted_disease}
- Confidence Level: {confidence_percent:.2f}%
"""
        
        if user_query:
            context += f"\nAdditional Context from Farmer: {user_query}"
        
        confidence_note = ""
        if confidence_percent < 60:
            confidence_note = f"\n**IMPORTANT**: The model confidence is {confidence_percent:.1f}%, which is relatively low. Please verify the diagnosis by checking the symptoms listed below carefully. Consider consulting an agricultural expert if symptoms don't match."
        
        full_prompt = f"""{self.agricultural_system_prompt}

{context}

Provide comprehensive information about "{predicted_disease}" following the exact format above with ALL sections.{confidence_note}

Write at least 300 words total. Be specific and detailed:"""
        
        try:
            response = self.model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def get_audio_response(self, transcribed_text):
        """
        Get response for audio-based query (after transcription)
        
        Args:
            transcribed_text: Text transcribed from audio
            
        Returns:
            Gemini's response as string
        """
        # Add context that this came from audio
        audio_context = f"""Voice query: {transcribed_text}

Provide a brief, clear answer:"""
        
        return self.get_text_response(audio_context)
    
    def get_general_farming_advice(self, topic):
        """
        Get general farming advice on a specific topic
        
        Args:
            topic: Topic of interest (e.g., "irrigation", "organic fertilizers")
            
        Returns:
            Gemini's response as string
        """
        prompt = f"""{self.agricultural_system_prompt}

Topic: {topic}

Provide key points only:"""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def format_response_for_display(self, response_text):
        """
        Format Gemini response for better display in Streamlit
        
        Args:
            response_text: Raw response from Gemini
            
        Returns:
            Formatted response
        """
        # The response is already well-formatted by Gemini
        # We can add additional formatting if needed
        return response_text


def test_gemini_helper():
    """Test the Gemini helper"""
    print("=" * 60)
    print("Gemini Helper Test")
    print("=" * 60)
    
    try:
        helper = GeminiHelper()
        
        # Test 1: Text query
        print("\nTest 1: Text Query")
        print("-" * 60)
        query = "My tomato plants have yellow leaves with brown spots. What should I do?"
        print(f"Query: {query}")
        print("\nResponse:")
        response = helper.get_text_response(query)
        print(response)
        
        # Test 2: Image diagnosis
        print("\n" + "=" * 60)
        print("Test 2: Image Diagnosis Response")
        print("-" * 60)
        predicted_disease = "Tomato Late Blight"
        confidence = 0.92
        print(f"Predicted Disease: {predicted_disease}")
        print(f"Confidence: {confidence * 100:.2f}%")
        print("\nResponse:")
        response = helper.get_image_diagnosis_response(predicted_disease, confidence)
        print(response)
        
        print("\n" + "=" * 60)
        print("Tests Complete!")
        print("=" * 60)
        
    except ValueError as e:
        print(f"\nError: {e}")
        print("\nPlease ensure you have set GEMINI_API_KEY in your .env file")
        print("Create a .env file with: GEMINI_API_KEY=your_api_key_here")


if __name__ == "__main__":
    test_gemini_helper()
