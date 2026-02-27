import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def create_page(title, lines, fontsize=12, start_y=0.85, line_spacing=0.06):
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.5, 0.93, title, ha='center', fontsize=fontsize + 4, fontweight='bold')
    
    y = start_y
    for text in lines:
        if text == "":
            y -= line_spacing * 0.5
            continue
        
        # Determine if it's an equation block (simple heuristic)
        if text.startswith("  $"):
            fig.text(0.15, y, text.strip(), ha='left', fontsize=fontsize + 2, color='blue')
        elif text.startswith("- "):
            fig.text(0.1, y, text, ha='left', fontsize=fontsize)
        else:
            fig.text(0.08, y, text, ha='left', fontsize=fontsize, wrap=True)
            
        y -= line_spacing
    return fig

def generate_pdf(filename="assignment_answers.pdf"):
    with PdfPages(filename) as pdf:
        # PAGE 1
        p1_lines = [
            "Part 1: System Design",
            "",
            "Q1. Max exposure time for < 2 px blur at 100 MPH",
            "- Ball Velocity v = 100 MPH = 44.704 m/s",
            "- Standard baseball diameter D = 9 in / pi = 2.86 in = 0.0726 m",
            "- Assuming conditions from Q2, ball occupies > 200 px:",
            "  Spatial Resolution (R) = 200 px / 0.0726 m = 2754.8 px/m",
            "  Image Velocity (v_px) = v * R = 44.704 * 2754.8 = 123,151 px/s",
            "",
            "We want max blur = 2 pixels. Thus:",
            "  $ t_{exp} <= 2 / v_{px} = 2 / 123151 = 1.624 x 10^{-5} s $",
            "  $ t_{exp} <= 16.24 \\mu s $  (or about 1/61,500 shutter speed)",
            "",
            "Q2. Sensor resolution & focal length (5m away, 200px diameter)",
            "- Distance Z = 5 m ; Diameter D = 0.0726 m ; Image d = 200 px",
            "- Using pinhole camera relation: d / f_px = D / Z",
            "  $ f_{px} = d * (Z / D) = 200 * (5 / 0.0726) = 13,774 px $",
            "",
            "- Assuming a standard pixel size p = 5 um/px (0.005 mm/px):",
            "  $ f = f_{px} * p = 13774 * 0.005 = 68.87 mm $",
            "- We need a focal length >= 69mm (e.g., standard 75mm lens).",
            "- Sensor Res: to track 2m flight path width -> 2m * 2754.8 px/m = 5510 px",
            "- Thus, a high-speed sensor like 6K or wide aspect ratio is needed.",
        ]
        fig1 = create_page("Answers", p1_lines)
        pdf.savefig(fig1)
        plt.close(fig1)
        
        # PAGE 2
        p2_lines = [
            "Q3. Hough Circle Transform vs YOLO (Deep Learning)",
            "- Hough Circle is deterministic, runs fast on CPUs, and is geometry-based.",
            "  * Pro: Computationally cheap, real-time edge device friendly.",
            "  * Con: Fails easily on motion blur, shadows, or if ball shape distorts.",
            "- YOLO/Seg is purely data-driven, learning semantic representations.",
            "  * Pro: Highly robust to blur, varied lighting, and background noise.",
            "  * Con: Requires GPU/NPU for real-time (< 5ms) processing, and",
            "         requires gathering/labeling thousands of baseball frames.",
            "",
            "Q4. Visual challenge of 'Bullet Spin' (Axis parallel to Camera)",
            "- Challenge: The 3D seam pattern doesn't 'rotate around' the visible",
            "  hemisphere of the ball. It only rotates 2D in the image plane.",
            "- Because depth cues (seams vanishing/appearing over the horizon)",
            "  are absent, estimating the true 3D spatial phase is ill-posed.",
            "- Logic Change: Instead of mapping seam translation across a 3D sphere,",
            "  we calculate the 2D angular rotation of the seam contour relative to",
            "  the 2D centroid of the ball.",
            "",
            "Part 3: Bonus Implementation",
            "Q5. 3D Coordinate Transformation Matrix",
            "- Transform from Ball-Local coords (P_B) to Camera-Reference (P_C).",
            "- Let R be the 3x3 rotation matrix of the ball.",
            "- Let t = [Tx, Ty, Tz]^T be the 3D position of the ball in Camera space.",
            "",
            "  $ P_C = R * P_B + t $",
            "",
            "In Homogeneous 4x4 Transformation Matrix form:",
            "  $ T = [ R_{3x3} , t_{3x1} ] $",
            "  $     [ 0_{1x3} ,   1   ] $",
        ]
        fig2 = create_page("Assignment Answers (Cont.)", p2_lines)
        pdf.savefig(fig2)
        plt.close(fig2)

if __name__ == "__main__":
    print("Generating PDF...")
    generate_pdf()
    print("Done generating assignment_answers.pdf")
