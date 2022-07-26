# MagnetoPyElastica Examples

This directory contains number of examples of magneto pyelastica.
Each [example cases](#example-cases) are stored in separate subdirectories, containing case descriptions, run file, and all other data/script necessary to run.

## Installing Requirements
In order to run examples, you will need to install additional dependencies.

```bash
poetry install -E examples
```

For making videos we are using the `ffmpeg` package. You will need to install it using `conda-forge`

```bash
conda install -c conda-forge ffmpeg
```

## Case Examples

Examples can serve as a starting template for customized usages.

* [ConstantMagnetiField](./ConstantMagneticField)
    * __Purpose__: Physical convergence test of simple magnetic rod under constant field.
    * __Features__: CosseratRod, MagneticForces, ConstantMagneticField
* [RotatingMagneticField](./RotatingMagneticField)
  * __Purpose__ : Magnetic rod  under rotating magnetic field.
  * __Features__: CosseratRod, MagneticForces, SingleModeOscillatingMagneticField
* [Magnetic2DCiliaCarpet](./Magnetic2DCiliaCarpet)
    * __Purpose__ : Many magnetic rods that have different magnetization direction under rotating magnetic field.
    * __Features__: CosseratRod, MagneticForces, SingleModeOscillatingMagneticField
