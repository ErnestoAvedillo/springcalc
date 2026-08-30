from springcalc.lineal.goodman import GoodmanAnalyzer, GoodmanData
import matplotlib.pyplot as plt
import numpy as np
import io
import base64


def generate_goodman_diagram(spring, initial_length, final_length, shot_peening=False, number_cycles=1e6):
    """Generate the Goodman diagram for the spring"""
    try:
        # Calculate the maximum and minimum stresses
        max_deflection = spring.free_length - final_length
        min_deflection = spring.free_length - initial_length

        max_load = spring.spring_constant * max_deflection if max_deflection > 0 else 0
        min_load = spring.spring_constant * min_deflection if min_deflection > 0 else 0

        # Shear stress using the Wahl formula
        max_stress = (8 * max_load * spring.mean_diameter * spring.wahl_factor) / (np.pi * spring.wire_diameter**3) if max_load > 0 else 0
        min_stress = (8 * min_load * spring.mean_diameter * spring.wahl_factor) / (np.pi * spring.wire_diameter**3) if min_load > 0 else 0

        # Create the Goodman analysis
        goodman_data = GoodmanData(
            material=spring.material,
            wire_diameter=spring.wire_diameter,
            load_type="torsion",  # For helical springs this is torsional load
            number_cycles=number_cycles
        )

        analyzer = GoodmanAnalyzer(goodman_data, shot_peening=shot_peening)

        # Generate the diagram
        fig = analyzer.plot_diagram(max_stress, min_stress, show_plot=False)

        # Save the diagram as base64
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        goodman_image = base64.b64encode(buffer.getvalue()).decode()
        plt.close()

        # Get the analysis summary
        analysis = analyzer.get_analysis_summary(max_stress, min_stress)

        return {
            'image': goodman_image,
            'analysis': analysis,
            'stresses': {
                'stress_max': round(max_stress, 2),
                'stress_min': round(min_stress, 2),
                'load_max': round(max_load, 2),
                'load_min': round(min_load, 2)
            }
        }
    except Exception as e:
        print(f"Error generating Goodman diagram: {e}")
        return None
