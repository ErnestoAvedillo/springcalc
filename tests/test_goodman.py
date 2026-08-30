from springcalc.lineal.goodman import Goodman, GoodmanData, GoodmanAnalyzer
from springcalc.regresiones.factor_f.usar_modelo_factor_f import ModelFactorF
from springcalc.pymodels.material import Material
import matplotlib.pyplot as plt
from pandas import read_csv
import json
import os

def test_goodman():
    """Full test of the Goodman functionality using the new architecture"""

    # Configure matplotlib to display plots
    plt.ion()  # Enable interactive mode

    print("=== GOODMAN ANALYSIS ===")
    print("Testing the new separated architecture...")

    # 1. Create the material data
    material = Material(material_name="DH")
    diameter = 1.0
    load_type = "torsion"

    # 2. Option A: use the new architecture (recommended)
    print("\n--- Method 1: New architecture ---")
    data = GoodmanData(material=material, wire_diameter=diameter, load_type=load_type, number_cycles=1e5)
    analyzer = GoodmanAnalyzer(data)

    # Define the operating point
    sigma_max = 400  # MPa
    sigma_min = 100  # MPa

    # Show the analysis info
    summary = analyzer.get_analysis_summary(sigma_max, sigma_min)
    print(f"Material: {summary['material']}")
    print(f"Diameter: {summary['wire_diameter']} mm")
    print(f"Load type: {summary['load_type']}")
    print(f"Safety factor: {summary['safety_factor']:.2f}")
    print(f"RMa min: {summary['strengths']['RMa_min_MPa']:.1f} MPa")
    print(f"Se: {summary['strengths']['Se_MPa']:.1f} MPa")
    print(f"Sf: {summary['strengths']['Sf_MPa']:.1f} MPa")

    # Show the improved diagram
    print("\nShowing the improved Goodman diagram...")
    fig1 = analyzer.plot_diagram(sigma_max, sigma_min, show_plot=True)

    # 3. Option B: use the backwards-compatibility class
    print("\n--- Method 2: Backwards compatibility ---")
    goodman_legacy = Goodman(material=material, wire_diameter=diameter, load_type=load_type)
    print("Testing the legacy plot_diagram method...")
    goodman_legacy.plot_diagram(sigma_max, sigma_min)

    # 4. Additional analysis with the new system
    print("\n--- Additional analysis ---")
    print(f"Factor kₐ (surface): {analyzer.k_a:.4f}")
    print(f"Factor k_b (size): {analyzer.k_b:.4f}")
    print(f"Factor k_c (load): {analyzer.k_c:.4f}")

    # Test with different operating points
    test_points = [
        (300, 50),   # Safe point
        (500, 200),  # Critical point
        (200, 0),    # Mean stress only
    ]

    print("\n--- Analysis of multiple points ---")
    for i, (s_max, s_min) in enumerate(test_points):
        sf = analyzer.calculate_safety_factor(s_max, s_min)
        status = "SAFE" if sf > 1.5 else "CRITICAL" if sf > 1.0 else "DANGEROUS"
        print(f"Point {i+1}: σmax={s_max}, σmin={s_min} → FS={sf:.2f} [{status}]")

    # Save the summary to JSON for later analysis.
    # default=str serializes pint Quantity objects (not natively serializable).
    with open('/tmp/goodman_analysis.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to: /tmp/goodman_analysis.json")

    plt.ioff()  # Disable interactive mode

def test_comparacion_materiales():
    """Additional test: compare different materials"""
    print("\n=== MATERIAL COMPARISON ===")

    materiales = ["DH", "DM"]  # Add more if available
    diameter = 1.5
    load_type = "torsion"
    sigma_max, sigma_min = 350, 80

    for mat_name in materiales:
        try:
            material = Material(material_name=mat_name)
            data = GoodmanData(material=material, wire_diameter=diameter, load_type=load_type, number_cycles=1e4)
            analyzer = GoodmanAnalyzer(data)
            sf = analyzer.calculate_safety_factor(sigma_max, sigma_min)

            print(f"{mat_name}: FS = {sf:.2f}, RMa = {analyzer.wire_char.RMa_min:.0f} MPa")

        except Exception as e:
            print(f"Error with material {mat_name}: {e}")

def test_factorf():
    """Specific test for the factor f model"""
    print("\n=== FACTOR F MODEL TEST ===")
    material = Material(material_name="DH")
    load_type = "torsion"
    import springcalc.material as material_pkg
    csv_path = os.path.join(os.path.dirname(material_pkg.__file__), "DIAMETRO_TOLERANCIAS.csv")
    pd = read_csv(csv_path)
    diameters = pd['diameter'].values
    valores_sf =[]
    for diameter in diameters:
        data = GoodmanData(material=material, wire_diameter=diameter, load_type=load_type, number_cycles=10000)
        analyzer = GoodmanAnalyzer(data)
        valores_sf.append(analyzer.Ssf)
        print(f"Diameter: {diameter:.2f} mm → Sf: {analyzer.Ssf:.1f} MPa")
    plt.figure(figsize=(8, 5))
    plt.plot(diameters, valores_sf, marker='o')
    plt.title("Fatigue strength Sf vs Wire Diameter")
    plt.xlabel("Wire diameter (mm)")
    plt.ylabel("Fatigue strength Sf (MPa)")
    plt.grid(True)
    plt.show()

    material = Material(material_name="DH")
    load_type = "torsion"
    diameters = [0.5, 1.0, 1.5, 2.0]
    cycles = [1e3, 1e4, 1e5, 1e6]
    markers = ['o', 's', 'D', '^']
    for diameter in diameters:
        valores_sf = []
        valores_se = []
        for cycle in cycles:
            data = GoodmanData(material=material, wire_diameter=diameter, load_type=load_type, number_cycles=cycle)
            analyzer = GoodmanAnalyzer(data)
            print(f"Diameter: {diameter:.2f} mm, Cycles: {cycle:.0f} → Sf: {analyzer.Ssf:.1f} MPa")
            valores_sf.append(analyzer.Ssf)
            valores_se.append(analyzer.Sse)
        plt.figure(figsize=(8, 5))
        plt.plot(cycles, valores_sf, marker=markers[diameters.index(diameter)], label=f'Diameter: {diameter:.2f} mm')
        plt.plot(cycles, valores_se, marker=markers[diameters.index(diameter)], label=f'Diameter: {diameter:.2f} mm')
        plt.xscale('log')
        plt.title(f"Fatigue strength Sf vs Cycles (Diameter: {diameter:.2f} mm)")
        plt.xlabel("Cycles (log scale)")
        plt.ylabel("Fatigue strength Sf (MPa)")
        plt.grid(True)
        plt.legend()
    plt.show()

if __name__ == "__main__":
    # test_goodman()
    # test_comparacion_materiales()
    test_factorf()
    input("Press enter...")
