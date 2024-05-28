# AtomEyes

AtomEyes is an open-source tool for waveform alignment using convolutional dictionary learning (CDL). It provides a suite of functions to analyze and visualize the temporal patterns and activations of atoms learned by a CDL model. This README provides an overview of the functionalities and usage examples.

## Features

- **Plot Temporal Patterns**: Visualize the temporal patterns of the atoms in a CDL model.
- **Event Coverage**: Display the event coverage of each atom.
- **Scatter Plot of Activations**: Generate scatter plots of atom activations across trials.
- **Scatter Plot of Extrema**: Visualize peaks and troughs in the activations.
- **Conditional Probability Density Function (CPDF)**: Plot the CPDF of peaks and troughs over time.
- **Comparison of Average Signal Segments**: Compare the average signal segments for each atom between different trial classes.
- **Align and Plot Signal Segments**: Align signal segments by extrema and plot the comparison between different trial classes.

## To-DO

- Parameters for "modes": peak/rise/decay/trough, burst detection, 
- Find optimal range of n_times_atom based on frequency band and mode
- Group cycles based on temporal distribution of extrema
- Limit atom timing in comparison (see above)
- Put plot save options in all functions or take out
- Compare with multiple atoms (plot_atom_comparison)
- Add jitter to data alignment to randomize
- Add clipping to data alignment

## Installation

To install AtomEyes, clone the repository and install the required dependencies:

```bash
git clone https://github.com/username/AtomEyes.git
cd AtomEyes
pip install -r requirements.txt
```

## Usage

### Plot Temporal Patterns

```from alphacsc import ConvolutionalDictionaryLearning
import matplotlib.pyplot as plt

# Assuming `cdl_model` is a fitted instance of ConvolutionalDictionaryLearning
plot_atoms(cdl_model, sfreq=256)
```

## Contributing

If you have questions about a specific dataset on which you'd like to try this tool, please reach out to david_zhou@brown.edu .

## License

AtomEyes is licensed under the MIT License.