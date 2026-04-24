import pandas as pd
import os
from dotenv import load_dotenv
import google.generativeai as genai
import time

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

persuade_df = pd.read_csv('data/raw/persuade_with_topics.csv')
model = genai.GenerativeModel('models/gemini-2.5-flash')
generated_essays = []

print(f"\n🚀 Generating 1,000 Gemini essays with MATCHED topics...")
print("This will take approximately 10-15 minutes.\n")

for idx in range(1000):
    human_essay = persuade_df.iloc[idx]['full_text']
    essay_length = len(human_essay)
    topic = persuade_df.iloc[idx]['extracted_topic']
    
    prompt = f"Write an argumentative essay about {topic} that is approximately {essay_length} characters long."
    
    try:
        response = model.generate_content(prompt)
        generated_text = response.text
        
        generated_essays.append({
            'essay_id': idx,
            'source_id': persuade_df.iloc[idx]['text_id'],
            'topic': topic,
            'target_length': essay_length,
            'generated_text': generated_text,
            'actual_length': len(generated_text)
        })
        
        if (idx + 1) % 50 == 0:
            print(f"✅ Generated {idx + 1}/1000 essays...")
        
        time.sleep(0.5)
        
    except Exception as e:
        print(f"❌ Error at essay {idx}: {e}")
        generated_essays.append({
            'essay_id': idx,
            'source_id': persuade_df.iloc[idx]['text_id'],
            'topic': topic,
            'target_length': essay_length,
            'generated_text': f"GENERATION_FAILED: {str(e)}",
            'actual_length': 0
        })

result_df = pd.DataFrame(generated_essays)
result_df.to_csv('data/raw/gemini_samples.csv', index=False)

print(f"\n✅ COMPLETE! Generated {len(result_df)} essays")
successful = len(result_df[~result_df['generated_text'].str.contains('FAILED', na=False)])
print(f"Success rate: {successful/len(result_df)*100:.1f}%")