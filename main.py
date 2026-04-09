from transformers import pipeline
import requests
from PIL import Image
from io import BytesIO
import time

# ============= Sentiment Analysis =============
print("1. Testing Sentiment Analysis...")
classifier = pipeline("sentiment-analysis")
result = classifier("This setup is working perfectly")
print(f"Sentiment: {result}\n")

# ============= Image Processing =============
print("2. Testing Image Processing...")
try:
    # Download a sample image
    img_url = "https://httpbin.org/image/png"
    response = requests.get(img_url, timeout=10)
    img = Image.open(BytesIO(response.content))
    print(f"Image downloaded: {img.format}, Size: {img.size}")
    img.save("test_image.png")
    print("Image saved as test_image.png\n")
except Exception as e:
    print(f"Image processing error: {e}\n")

# ============= Text Generation =============
print("3. Testing Text Generation...")
try:
    generator = pipeline("text-generation", model="distilgpt2")
    gen_result = generator("Machine learning is", max_length=50, num_return_sequences=1)
    print(f"Generated: {gen_result[0]['generated_text']}\n")
except Exception as e:
    print(f"Text generation error: {e}\n")

# ============= Named Entity Recognition =============
print("4. Testing Named Entity Recognition...")
try:
    ner_pipeline = pipeline("ner")
    text = "Apple Inc. was founded by Steve Jobs in Cupertino, California."
    ner_result = ner_pipeline(text)
    print(f"Text: {text}")
    print(f"Entities found: {len(ner_result)}")
    for entity in ner_result[:3]:
        print(f"   - {entity['word']}: {entity['entity']}")
    print()
except Exception as e:
    print(f"NER error: {e}\n")

# ============= Model Testing Utils =============
print("5. Model Testing Utils:")

def benchmark_model(pipeline_name, input_text, iterations=3):
    """Benchmark a model's inference speed"""
    try:
        pipe = pipeline(pipeline_name)
        times = []
        for i in range(iterations):
            start = time.time()
            _ = pipe(input_text)
            times.append(time.time() - start)
        avg_time = sum(times) / len(times)
        print(f"   {pipeline_name}: avg {avg_time:.3f}s ({iterations} iterations)")
        return avg_time
    except Exception as e:
        print(f"   {pipeline_name}: Error - {e}")
        return None

def test_pipeline(name, pipe_type, test_input):
    """Generic pipeline tester"""
    try:
        pipe = pipeline(pipe_type)
        result = pipe(test_input)
        print(f"   ✓ {name} works")
        return result
    except Exception as e:
        print(f"   ✗ {name} failed: {e}")
        return None

# Run benchmarks
benchmark_model("sentiment-analysis", "This is a test sentence")

print("\nML testing utilities ready!")