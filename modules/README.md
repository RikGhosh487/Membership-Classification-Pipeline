# Custom Python Modules
Contains custom designed Python modules that are used by [main.py](https://github.com/RikGhosh487/Membership-Classification-Pipeline/blob/main/main.py) to progress through each step of the primary pipeline followed to distinguish possible cluster members from non member field data points. These modules perform **READ AND WRITE** operations on the original data supplied to [main.py](https://github.com/RikGhosh487/Membership-Classification-Pipeline/blob/main/main.py).

## Modules
| Name | Version | Description |
| --: | :--: | :-- |
| [db.py](https://github.com/RikGhosh487/Membership-Classification-Pipeline/blob/main/modules/db.py) | `v2.0.5` | Uses DBSCAN to remove remaining noise points from the dataset |
| [gamma_err.py](https://github.com/RikGhosh487/Membership-Classification-Pipeline/blob/main/modules/gamma_err.py) | `v1.0.2` | Applies a Gamma Distribution to the histogram and retains data based on a **user-supplied** threshold (please see [inputs.ini](https://github.com/RikGhosh487/Membership-Classification-Pipeline/blob/main/inputs.ini)) |
| [mixmod.py](https://github.com/RikGhosh487/Membership-Classification-Pipeline/blob/main/modules/mixmod.py) | `v3.1.1` | Uses Gaussian Mixture Models to split a certain reduced and normalized dataset into two gaussian bins (one for **members** and one for **non-members**). Only retains members |
| [two_tail_err.py](https://github.com/RikGhosh487/Membership-Classification-Pipeline/blob/main/modules/two_tail_err.py) | `v1.0.2` | Applies a Normal Distribution to the histogram and retains data based on a **user-defined** factor for the *standard deviation* away from the *mean* (please see [inputs.ini](https://github.com/RikGhosh487/Membership-Classification-Pipeline/blob/main/inputs.ini)) |
| [visualize.py](https://github.com/RikGhosh487/Membership-Classification-Pipeline/blob/main/modules/visualize.py) | `v1.9.2` | Displays the different fields of the data using `matplotlib` and `seaborn` |

### Express Links and Documentations
- [Matplotlib](https://matplotlib.org/) → Used for most of the visualization and plotting
- [NumPy](https://numpy.org/) → Used for numeric computations and faster array creations
- [SciKit-Learn](https://scikit-learn.org/stable/) → Primary Machine Learning and Data Science package for the entire pipeline
- [Kneed](https://kneed.readthedocs.io/en/stable/) → Subordinate package for DBSCAN
- [SciPy](https://scipy.org/) → Used for statistical computations
- [Pandas](https://pandas.pydata.org/) → Used for dataframes
- [Seaborn](https://seaborn.pydata.org/) → Used for advanced and prettier visualizations