# MagnetoPyElastica Examples

This directory contains number of examples illustrating usage of MagnetoPyElastica.
Each of the [example cases](#example-cases) is stored in separate subdirectories, containing case descriptions, run file, and all other data/scripts necessary to run.

## Installing Requirements
In order to run examples, you will need to install additional dependencies via

```bash
make install_examples_dependencies
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
