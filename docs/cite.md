# Citing NetKet

```{note}
NetKet provides a command-line tool `netket-cite` to help you generate the appropriate references and acknowledgment text.
```

NetKet is developed by several researchers, both as part of their work and in their spare time. 
As with anyone in academia, careers depend heavily on citations and publication impact. 
Acknowledging those contributions is important, as it encourages the growth of an high-quality open-source ecosystem where researchers are rewarded for taking an active effort.

```{important}
**If you use NetKet for academic research, or have used NetKet as inspiration or for learning in academic research, you must properly acknowledge it by citing BOTH NetKet publications.**
Additionally, depending on the algorithms you employed, you might have to cite additional algorithmic papers (read below). 

While some researchers still pretend that software citations are not due, such behaviour is deeply irrespective of the work put in by all contributors to NetKet (and other projects) and, quite frankly, offensive.
```

For proper acknowledgment, include a sentence, depending on your situation, like one of the following ones:

```latex
Simulations for this work were performed using 
NetKet~\cite{netket3:2022,netket2:2019}.
% or
Simulations for this work were performed 
with codes built on top of NetKet~\cite{netket3:2022,netket2:2019}.
% or
We acknowledge the NetKet codebase as a learning 
and inspiration resource for the codes used as 
part of this work~\cite{netket3:2022,netket2:2019}.

% additionally
This software is built on top of JAX~\cite{jax2018github} 
and Flax~\cite{flax2020github}.
```

<details class="citation-button">
<summary><strong>Click here to reveal the corresponding bibtex entries.</strong></summary>

```bibtex
@Article{netket3:2022,
    title={NetKet 3: Machine Learning Toolbox for Many-Body Quantum Systems},
    author={Filippo Vicentini and Damian Hofmann and Attila Szabó and Dian Wu and Christopher Roth and Clemens Giuliani and Gabriel Pescia and Jannes Nys and Vladimir Vargas-Calderón and Nikita Astrakhantsev and Giuseppe Carleo},
    journal={SciPost Phys. Codebases},
    pages={7},
    year={2022},
    publisher={SciPost},
    doi={10.21468/SciPostPhysCodeb.7},
    url={https://scipost.org/10.21468/SciPostPhysCodeb.7}
}

@article{netket2:2019,
    title={NetKet: A machine learning toolkit for many-body quantum systems},
    author={Carleo, Giuseppe and Choo, Kenny and Hofmann, Damian and Smith, James ET and Westerhout, Tom and Alet, Fabien and Davis, Emily J and Efthymiou, Stavros and Glasser, Ivan and Lin, Sheng-Hsuan and Mauri, Marta and Mazzola, Guglielmo and Pereira, Christian B and Vicentini, Filippo},
    journal={SoftwareX},
    volume={10},
    pages={100311},
    year={2019},
    publisher={Elsevier},
    doi={10.1016/j.softx.2019.100311},
    url={https://www.sciencedirect.com/science/article/pii/S2352711019300974}
}

@misc{jax2018github,
  title = {{{JAX}}: Composable Transformations of {{Python}}+{{NumPy}} Programs},
  author = {Bradbury, James and Frostig, Roy and Hawkins, Peter and Johnson, Matthew James and Leary, Chris and Maclaurin, Dougal and Necula, George and Paszke, Adam and VanderPlas, Jake and {Wanderman-Milne}, Skye and Zhang, Qiao},
  year = {2018}
}

@misc{flax2020github,
  title = {Flax: {{A}} Neural Network Library and Ecosystem for {{JAX}}},
  author = {Heek, Jonathan and Levskaya, Anselm and Oliver, Avital and Ritter, Marvin and Rondepierre, Bertrand and Steiner, Andreas and {van Zee}, Marc},
  year = {2024}
}

```
</details>

Several algorithms implemented in NetKet are derived from academic publications unrelated to NetKet itself. 
Those contributions should also be acknowledged.
NetKet provides a command-line tool `netket-cite` to help you generate the appropriate references and acknowledgment text.
You can also find an automatically-generated list of references below in the section [Algorithm-Specific Citations](#algorithm-specific-citations)



## Citation Tools

Use the `netket-cite` command-line tool to get the complete citation information:

```bash
# Display all relevant citations for your NetKet usage
netket-cite

# Generate a complete references.bib file
netket-cite --bib {optional_name.bib}

# From within python
>>> nk.cite(bib=True)
```

## Additional references

### MPI Support

If you use NetKet's MPI functionality, please also cite the mpi4jax library that enables this feature:

```bibtex
@article{mpi4jax:2021,
    title={mpi4jax: Zero-copy MPI communication of JAX arrays},
    author={Häfner, Dion and Vicentini, Filippo},
    journal={Journal of Open Source Software},
    volume={6},
    number={65},
    pages={3419},
    year={2021},
    publisher={The Open Journal},
    doi={10.21105/joss.03419},
    url={https://joss.theoj.org/papers/10.21105/joss.03419}
}
```

### Algorithm-Specific Citations

Many algorithms implemented in NetKet are based on original research publications. When using specific features or algorithms, cite the relevant papers that introduced these methods. The citation tools help identify these references.


```{netket-citations}
:format: detailed
```