# Physics-Informed Latency Prediction App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://physics-informed-latency-pred.streamlit.app/)

An interactive web application demonstrating physics-informed uncertainty discovery in network latency prediction. This app compares data-driven approaches with physics-based models for predicting network latency over geographical distances.

## Overview

Network latency prediction is crucial for network optimization and performance monitoring. This application showcases how incorporating physical laws (speed of light in optical fibers) can improve prediction accuracy and uncertainty quantification compared to purely data-driven approaches.

## Key Features

### Interactive Analysis
- **Real-time parameter adjustment** through sidebar controls
- **Confidence level tuning** (80%-99%)
- **Anomaly detection threshold** customization
- **Cross-validation** results display

### Uncertainty Discovery
- **Physics-informed uncertainty estimation** based on optical fiber properties
- **Extrapolation-aware data uncertainty** using bootstrap methods
- **Adaptive uncertainty quantification** for different prediction scenarios
- **Uncertainty-weighted anomaly detection**

### Model Comparison
- **Physics-based predictions** using fundamental constants
- **Data-driven linear regression** models
- **Performance metrics** (MSE, R², Precision, Recall, F1-Score)
- **Residual analysis** and visualization

## Scientific Foundation

### Physics Model
The application uses the fundamental speed of light in optical fiber:
- **Fiber speed**: 2×10⁸ m/s (≈67% of speed of light in vacuum)
- **True physics slope**: 0.005 ms/km
- Based on electromagnetic wave propagation in silica-based optical fibers

### Uncertainty Estimation Methods
1. **Blind Uncertainty Estimator**: Discovers data patterns without prior knowledge
2. **Adaptive Physics Uncertainty**: Adjusts uncertainty based on distance and physical constraints
3. **Bootstrap Uncertainty**: Statistical resampling for data-driven confidence intervals
4. **Extrapolation-Aware Methods**: Higher uncertainty for predictions outside training range

## Application Structure

The app is organized into three main tabs:

### Main Demo
- Model performance comparison
- Prediction visualizations
- Anomaly detection results
- Interactive parameter controls

### Uncertainty Analysis
- Detailed uncertainty quantification
- Pattern discovery results
- Confidence interval visualizations
- Method comparison tables

### Summary
- Production-ready uncertainty strategy
- Key insights and recommendations
- Performance summary metrics
- Best practices for deployment

## Technical Stack

- **Frontend**: Streamlit
- **ML/Stats**: scikit-learn, NumPy, pandas
- **Visualization**: Matplotlib, custom CSS styling
- **Data**: Enhanced simulation datasets with realistic network measurements

## Getting Started

### Prerequisites
```bash
pip install streamlit pandas scikit-learn numpy matplotlib
```

### Running Locally
```bash
git clone [your-repo-url]
cd physics-informed-latency-prediction
streamlit run app_driver.py
```

### Data Requirements
The application expects two CSV files:
- `data/enahnced_simulation_train_data.dat`
- `data/enahnced_simulation_test_data.dat`

Required columns:
- `geo_distance_km`: Geographic distance in kilometers
- `measured_latency_ms`: Actual measured latency in milliseconds
- `physics_latency_ms`: Physics-based predicted latency
- `is_anomaly`: Boolean flag for anomalous measurements

## Key Insights

### When to Use Physics-Informed Models
- **High uncertainty** in data-scarce regions
- **Extrapolation** beyond training data range
- **Physical constraints** need to be respected
- **Interpretability** is crucial for decision-making

### Uncertainty Benefits
- **Better anomaly detection** through uncertainty weighting
- **Confidence-aware predictions** for risk assessment
- **Adaptive model selection** based on uncertainty levels
- **Robust extrapolation** with quantified confidence

## Performance Metrics

The app provides comprehensive evaluation including:
- **Regression Metrics**: MSE, R², residual analysis
- **Classification Metrics**: Precision, Recall, F1-Score for anomaly detection
- **Uncertainty Metrics**: Coverage probability, uncertainty calibration
- **Cross-Validation**: K-fold validation with confidence intervals

## Use Cases

### Network Operations
- **Latency prediction** for route optimization
- **Anomaly detection** in network performance
- **Capacity planning** with uncertainty bounds
- **SLA compliance** monitoring

### Research Applications
- **Physics-informed ML** methodology demonstration
- **Uncertainty quantification** in networking
- **Hybrid modeling** approaches
- **Performance benchmarking**

## Customization

### Styling
The app uses custom CSS for professional appearance:
- Modern tab design with hover effects
- Color-coded insights boxes
- Responsive layout for different screen sizes

### Parameters
Adjustable through sidebar:
- **Confidence Level**: Statistical confidence for intervals
- **Anomaly Threshold**: Sensitivity for outlier detection
- **Validation Options**: Cross-validation and detailed analysis toggles

## Future Enhancements

- **Real-time data integration**
- **Multiple physics models** (satellite, wireless)
- **Advanced uncertainty methods** (Bayesian neural networks)
- **Interactive map visualization**
- **Model deployment** pipeline

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and documentation
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this application in your research, please cite:

```bibtex
@software{physics_informed_latency_pred,
  title={Physics-Informed Latency Prediction with Uncertainty Discovery},
  author={[Suchita Kulkarni]},
  year={2025},
  url={https://physics-informed-latency-pred.streamlit.app/}
}
```

## Contact

For questions, suggestions, or collaboration opportunities, please open an issue on GitHub or contact [suchita.kulkarni@gmail.com].

---
