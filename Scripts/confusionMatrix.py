import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

# Importing from your existing architecture
from RGBWithFFTDataset import RGBWithFFTDataset
from BlumMitchellCoTraining import BlumMitchellCoTraining
from helper_functions import serialize_confusion_matrix
# Adjust "main_script" if your original file has a different name
from cotraining import initialize_rgb_model, initialize_fft_model 

def matrices_to_anonymous_coded_latex(rgb_cm, fft_cm, class_names):
    """
    Generates a highly compressed side-by-side LaTeX table format.
    Replaces long class names with C1, C2, etc., and shifts the full names
    directly into the table caption as a legend dynamically.
    """
    num_classes = len(class_names)
    cols_format = f"l|{'c'*num_classes}"
    
    # Create the codes (C1, C2, ...) and the caption legend string
    codes = [f"C{i+1}" for i in range(num_classes)]
    legend_elements = [f"C{i+1} = {name}" for i, name in enumerate(class_names)]
    caption_legend = ", ".join(legend_elements)
    
    headers = " & ".join([f"\\textbf{{{c}}}" for c in codes])
    
    latex_str =  "% ---- Ultra-Narrow Coded Side-by-Side Tables ----\n"
    latex_str += "\\begin{table}[htbp]\n"
    latex_str += "  \\centering\n"
    latex_str += f"  \\caption{{Confusion Matrices for RGB (Left) and FFT (Right) Models. Legend: {caption_legend}.}}\n"
    latex_str += "  \\label{tab:compact_backbone_comparison}\n"
    
    # --- RGB Tabular ---
    latex_str += "  \\resizebox{0.47\\textwidth}{!}{%\n"
    latex_str += f"    \\begin{{tabular}}{{{cols_format}}}\n"
    latex_str += "      \\hline\n"
    latex_str += f"      \\textbf{{Act $\\backslash$ Pred}} & {headers} \\\\\n"
    latex_str += "      \\hline\n"
    for i in range(num_classes):
        row_values = [str(int(rgb_cm[i, j])) for j in range(num_classes)]
        latex_str += f"      \\textbf{{{codes[i]}}} & " + " & ".join(row_values) + " \\\\\n"
    latex_str += "      \\hline\n"
    latex_str += "    \\end{tabular}\n"
    latex_str += "  }\n"
    
    latex_str += "  \\hfill\n"
    
    # --- FFT Tabular ---
    latex_str += "  \\resizebox{0.47\\textwidth}{!}{%\n"
    latex_str += f"    \\begin{{tabular}}{{{cols_format}}}\n"
    latex_str += "      \\hline\n"
    latex_str += f"      \\textbf{{Act $\\backslash$ Pred}} & {headers} \\\\\n"
    latex_str += "      \\hline\n"
    for i in range(num_classes):
        row_values = [str(int(fft_cm[i, j])) for j in range(num_classes)]
        latex_str += f"      \\textbf{{{codes[i]}}} & " + " & ".join(row_values) + " \\\\\n"
    latex_str += "      \\hline\n"
    latex_str += "    \\end{tabular}\n"
    latex_str += "  }\n"
    
    latex_str += "\\end{table}\n"
    return latex_str

def generate_confusion_matrices():
    # 1. Define exact paths
    test_path = r"C:\Users\Lavinia\Desktop\@BejanRobert\cotraining_dissertation\small_labeled_ultrasound_dataset\test"
    model_fft_path = r"C:\Users\Lavinia\Desktop\@BejanRobert\cotraining_dissertation\models\small_80_start5_rgb0.95_fft0.9_fft2.pth"
    model_rgb_path = r"C:\Users\Lavinia\Desktop\@BejanRobert\cotraining_dissertation\models\small_80_start5_rgb0.95_fft0.9_rgb2.pth"

    input_size_rgb = (227, 227)
    input_size_fft = (224, 224)
    batch_size = 30

    # 2. Transforms
    rgb_transform = transforms.Compose([
        transforms.Resize(input_size_rgb),
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Normalize([0.5], [0.2])
    ])

    fft_transform = transforms.Compose([
        transforms.Resize(input_size_fft),
        transforms.Lambda(lambda x: x.unsqueeze(0) if x.dim() == 2 else x),
        transforms.Normalize([0.5], [0.2])
    ])

    # 3. Load dataset & extract classes
    print("Loading test dataset...")
    test_dataset = RGBWithFFTDataset(test_path, rgb_transform, fft_transform, labeled=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Dynamic Class Extraction
    class_names = test_dataset.classes
    num_classes = len(class_names)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    print(f"Detected {num_classes} classes: {class_names}")

    # 4. Initialize and load weights
    model_rgb = initialize_rgb_model(num_classes, device)
    model_fft = initialize_fft_model(num_classes, device)

    print("Loading pre-trained weights...")
    model_rgb.load_state_dict(torch.load(model_rgb_path, map_location=device))
    model_fft.load_state_dict(torch.load(model_fft_path, map_location=device))

    # 5. Setup evaluation wrapper
    cotrainer = BlumMitchellCoTraining(
        model_rgb, model_fft, num_classes, device,
        cotraining_start=5, k=200, confidence_thresh_fft=0.9, confidence_thresh_rgb=0.95, checked_number=100
    )

    # 6. Run evaluation
    print("\nEvaluating models...")
    rgb_acc, fft_acc, combined_acc, rgb_cm, fft_cm, combined_cm = cotrainer.evaluate(test_loader)

    # 7. Standard Console Output with Class Names
    print("\n" + "="*50)
    print(f"Test Accuracy - RGB: {rgb_acc:.4f} | FFT: {fft_acc:.4f} | Combined: {combined_acc:.4f}")
    print("="*50 + "\n")

    print(f"Class layout ordering: {class_names}\n")

    # 8. Generate and Print LaTeX Code block structures
    print("=================== GENERATED LATEX CODES ===================\n")
    
    print(matrices_to_anonymous_coded_latex(rgb_cm, fft_cm, class_names))

if __name__ == "__main__":
    generate_confusion_matrices()