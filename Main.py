from vision import analyze_image
from reasoning import analyze_risk

def run_pipeline(image_path):
    print("🔍 Step 1: Image Analysis...")
    vision_output = analyze_image(image_path)
    print("Vision JSON:", vision_output)

    print("\n🧠 Step 2: Risk Reasoning...")
    final_output = analyze_risk(vision_output)
    print("Final Output:", final_output)

if __name__ == "__main__":
    run_pipeline("test.jpg")
