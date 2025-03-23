import os
import time
import subprocess
import argparse


def run_command(command, description=None):
    """
    Run a shell command and print the output
    
    Args:
        command (str): Command to run
        description (str): Description of the command
    """
    if description:
        print(f"\n{'='*50}\n{description}\n{'='*50}")
    
    print(f"\nExecuting command: {command}")
    start_time = time.time()
    
    # Run the command
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    # Print output in real-time
    for line in process.stdout:
        print(line, end='')
    
    # Wait for the process to complete
    process.wait()
    
    elapsed_time = time.time() - start_time
    print(f"\nCommand completed in {elapsed_time:.2f} seconds with exit code: {process.returncode}")
    
    return process.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Test the full multimodal sentiment analysis system")
    parser.add_argument("--skip_training", action="store_true", help="Skip training and use existing model")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs for training")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    success = True
    
    # Training step
    if not args.skip_training:
        train_cmd = f"python train.py --data_path dataset/LabeledText.xlsx --images_dir dataset --epochs {args.epochs} --batch_size {args.batch_size} --freeze_bert --freeze_resnet --force_cpu --text_only_mode"
        success = run_command(train_cmd, "TRAINING THE MODEL") and success
    
    # Test the interactive tool with sample texts
    if success:
        sample_texts = [
            "I really enjoyed this movie, it was fantastic!",
            "The weather is not very good today.",
            "This product is average, neither good nor bad."
        ]
        
        for text in sample_texts:
            inference_cmd = f'python run_sentiment_analysis.py --text "{text}" --checkpoint checkpoints/best_model.pth --force_cpu'
            success = run_command(inference_cmd, f"ANALYZING SENTIMENT: {text}") and success
    
    # Final status
    if success:
        print("\n" + "="*50)
        print("TEST COMPLETED SUCCESSFULLY!")
        print("="*50)
        print("\nYou can now use the sentiment analysis tool with your own text:")
        print("python run_sentiment_analysis.py")
    else:
        print("\n" + "="*50)
        print("TEST FAILED")
        print("="*50)
        print("\nPlease check the error messages above and fix any issues.")


if __name__ == "__main__":
    main() 