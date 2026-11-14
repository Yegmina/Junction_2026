#!/usr/bin/env python3
"""Build an explicitly experimental 3D campaign from pinned DuMux sources.

The generated C++ sources retain the upstream GPL-3.0-or-later headers and are
written beside the upstream test, separately from the unmodified regression.
This custom application is not numerically/physically accepted merely because
the upstream regression passed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/media/yehormsi/External SSD Volume/AeroRozum-thermal-rl/physics/dumux'))
    p.add_argument('--jobs',type=int,default=4)
    p.add_argument('--cached',action='store_true',help='Enable upstream geometry/state/flux caches; equations and tolerances unchanged')
    p.add_argument('--ledger',action='store_true',help='Record exact DuMux storage/boundary/target flux budgets')
    p.add_argument('--surface',action='store_true',help='Use reconstructed face temperature, consistent water-component enthalpy and endpoint forcing')
    p.add_argument('--graded',action='store_true',help='Use upstream tensor-product grid manager for concentrated resolution')
    p.add_argument('--cached-parameters',action='store_true',help='Cache immutable physical parameters, without changing values/equations')
    p.add_argument('--compact-output',action='store_true',help='Every solved surface frame as float64; hourly interior VTK checkpoints')
    p.add_argument('--amgx',action='store_true',help='Build explicit optional in-process fp64 AMGX adapter, CPU default retained')
    p.add_argument('--axisymmetric',action='store_true',help='Separate native rotational-extrusion candidate for a single vertical cylinder in homogeneous soil')
    p.add_argument('--mpi-compact',action='store_true',help='Gather only owned surface faces across native DuMux MPI ranks')
    args=p.parse_args();real=args.root
    if args.compact_output and not (args.surface and args.ledger):raise ValueError('Compact output requires surface/ledger variants')
    if args.axisymmetric and not (args.graded and args.cached_parameters and args.compact_output):raise ValueError('Axisymmetric candidate requires graded/parameter-cached/compact variants')
    state=json.loads((real/'status.json').read_text())
    if state.get('status')!='upstream_test_passed':raise RuntimeError('Pinned upstream regression must pass before custom application build')
    root=Path('/tmp')/('aerorozum-dumux-'+hashlib.sha256(str(real).encode()).hexdigest()[:12])
    if not root.exists():root.symlink_to(real,target_is_directory=True)
    src=root/'dumux';base=src/'test/porousmediumflow/richards/nonisothermal'
    suffix=('_cached' if args.cached else '')+('_ledger' if args.ledger else '')+('_surface' if args.surface else '')+('_graded' if args.graded else '')+('_params' if args.cached_parameters else '')+('_compact' if args.compact_output else '')+('_amgx' if args.amgx else '')
    app_name='aerorozum_campaign'+suffix
    binary_name='aerorozum_richardsni_campaign'+suffix
    file_suffix=('-cached' if args.cached else '')+('-ledger' if args.ledger else '')+('-surface' if args.surface else '')+('-graded' if args.graded else '')+('-params' if args.cached_parameters else '')+('-compact' if args.compact_output else '')+('-amgx' if args.amgx else '')
    if args.axisymmetric:suffix+='_axisymmetric';app_name='aerorozum_campaign'+suffix;binary_name='aerorozum_richardsni_campaign'+suffix;file_suffix+='-axisymmetric'
    if args.mpi_compact:suffix+='_mpi';app_name='aerorozum_campaign'+suffix;binary_name='aerorozum_richardsni_campaign'+suffix;file_suffix+='-mpi'
    app=base/app_name
    app.mkdir(exist_ok=True)
    def replace_exact(text,old,new):
        if old not in text:raise RuntimeError('Pinned upstream source does not match expected patch: '+old[:80])
        return text.replace(old,new)
    properties=(base/'evaporation/properties.hh').read_text()
    properties=replace_exact(properties,'#include "../spatialparams.hh"','#include "spatialparams.hh"')
    properties=replace_exact(properties,'Dune::YaspGrid<2>','Dune::YaspGrid<3>')
    if args.graded:properties=replace_exact(properties,'Dune::YaspGrid<3>','Dune::YaspGrid<3,Dune::TensorProductCoordinates<GetPropType<TypeTag,Properties::Scalar>,3>>')
    if args.cached:
        properties=replace_exact(properties,'} // end namespace Dumux::Properties','''
template<class TypeTag> struct EnableGridGeometryCache<TypeTag,TTag::RichardsNIEvaporation> { static constexpr bool value=true; };
template<class TypeTag> struct EnableGridVolumeVariablesCache<TypeTag,TTag::RichardsNIEvaporation> { static constexpr bool value=true; };
template<class TypeTag> struct EnableGridFluxVariablesCache<TypeTag,TTag::RichardsNIEvaporation> { static constexpr bool value=true; };
} // end namespace Dumux::Properties''')
    if args.axisymmetric:
        properties=replace_exact(properties,'Dune::YaspGrid<3,Dune::TensorProductCoordinates<GetPropType<TypeTag,Properties::Scalar>,3>>','Dune::YaspGrid<2,Dune::TensorProductCoordinates<GetPropType<TypeTag,Properties::Scalar>,2>>')
        properties=replace_exact(properties,'#include <dune/grid/yaspgrid.hh>','#include <dune/grid/yaspgrid.hh>\n#include <dumux/discretization/extrusion.hh>')
        properties=replace_exact(properties,'} // end namespace Dumux::Properties','''
template<class TypeTag> struct GridGeometry<TypeTag,TTag::RichardsNIEvaporationCC>
{
    using GridView=typename GetPropType<TypeTag,Properties::Grid>::LeafGridView;
    struct Traits : public CCTpfaDefaultGridGeometryTraits<GridView> { using Extrusion=RotationalExtrusion<0>; };
    using type=CCTpfaFVGridGeometry<GridView,getPropValue<TypeTag,Properties::EnableGridGeometryCache>(),Traits>;
};
} // end namespace Dumux::Properties''')
    (app/'properties.hh').write_text(properties)
    spatial=(base/'spatialparams.hh').read_text()
    spatial=replace_exact(spatial,'return permeability_;','return inTarget_(globalPos) ? getParam<Scalar>("Target.Permeability", 1e-24) : permeability_;')
    spatial=replace_exact(spatial,'return porosity_;','return inTarget_(globalPos) ? getParam<Scalar>("Target.Porosity", 1e-6) : porosity_;')
    spatial=replace_exact(spatial,'permeability_ = 1e-10;','permeability_ = getParam<Scalar>("SpatialParams.Permeability", 2e-12);')
    spatial=replace_exact(spatial,'porosity_ = 0.4;','porosity_ = getParam<Scalar>("SpatialParams.Porosity", 0.43);')
    additions='''
    template<class E, class S, class ES, class SS>
    Scalar solidThermalConductivity(const E&, const S& scv, const ES&, const SS&) const
    { return inTarget_(scv.dofPosition()) ? getParam<Scalar>("Target.ThermalConductivity", .2) : getParam<Scalar>("Component.SolidThermalConductivity", 2.8); }
    template<class E, class S, class ES, class SS>
    Scalar solidHeatCapacity(const E&, const S& scv, const ES&, const SS&) const
    { return inTarget_(scv.dofPosition()) ? getParam<Scalar>("Target.HeatCapacity", 1200.) : getParam<Scalar>("Component.SolidHeatCapacity", 790.); }
    template<class E, class S, class ES, class SS>
    Scalar solidDensity(const E&, const S& scv, const ES&, const SS&) const
    { return inTarget_(scv.dofPosition()) ? getParam<Scalar>("Target.Density", 1400.) : getParam<Scalar>("Component.SolidDensity", 2700.); }
'''
    spatial=replace_exact(spatial,'private:\n',additions+'''\nprivate:
    bool inTarget_(const GlobalPosition& pos) const
    {
        if (!getParam<bool>("Target.Enabled", true)) return false;
        const auto upper = getParam<GlobalPosition>("Grid.UpperRight");
        const Scalar dx=pos[0]-getParam<Scalar>("Target.X",.5);
        const Scalar dy=pos[1]-getParam<Scalar>("Target.Y",.5);
        const Scalar depth=upper[2]-pos[2];
        const Scalar top=getParam<Scalar>("Target.Depth",.03);
        const Scalar radius=getParam<Scalar>("Target.Radius",.12);
        return dx*dx+dy*dy <= radius*radius && depth>=top && depth<=top+getParam<Scalar>("Target.Thickness",.05);
    }
''')
    if args.cached_parameters:
        values={
            'Target.X':('targetX_', '.5'),'Target.Y':('targetY_', '.5'),
            'Target.Depth':('targetDepth_', '.03'),'Target.Radius':('targetRadius_', '.12'),
            'Target.Thickness':('targetThickness_', '.05'),'Target.Permeability':('targetPermeability_', '1e-24'),
            'Target.Porosity':('targetPorosity_', '1e-6'),'Target.ThermalConductivity':('targetConductivity_', '.2'),
            'Target.HeatCapacity':('targetHeatCapacity_', '1200.'),'Target.Density':('targetDensity_', '1400.'),
            'Component.SolidThermalConductivity':('soilConductivity_', '2.8'),
            'Component.SolidHeatCapacity':('soilHeatCapacity_', '790.'),'Component.SolidDensity':('soilDensity_', '2700.')}
        for key,(member,default) in values.items():
            spatial=re.sub(r'getParam<Scalar>\("'+re.escape(key)+r'",\s*[^)]*\)',member,spatial)
        spatial=replace_exact(spatial,'getParam<bool>("Target.Enabled", true)','targetEnabled_')
        spatial=replace_exact(spatial,'getParam<GlobalPosition>("Grid.UpperRight")','bounds_')
        initializers='\n        targetEnabled_=getParam<bool>("Target.Enabled",true);\n        bounds_=getParam<GlobalPosition>("Grid.UpperRight");\n'
        initializers+=''.join(f'        {member}=getParam<Scalar>("{key}",{default});\n' for key,(member,default) in values.items())
        spatial=replace_exact(spatial,'porosity_ = getParam<Scalar>("SpatialParams.Porosity", 0.43);','porosity_ = getParam<Scalar>("SpatialParams.Porosity", 0.43);'+initializers)
        declarations='\n    bool targetEnabled_;\n    GlobalPosition bounds_;\n'+''.join(f'    Scalar {member};\n' for member,_ in values.values())
        spatial=replace_exact(spatial,'    const PcKrSwCurve pcKrSwCurve_;',declarations+'\n    const PcKrSwCurve pcKrSwCurve_;')
    if args.axisymmetric:
        start=spatial.index('    bool inTarget_(const GlobalPosition& pos) const');end=spatial.index('\n    }',start)+len('\n    }')
        spatial=spatial[:start]+'''    bool inTarget_(const GlobalPosition& pos) const
    {
        const Scalar below=bounds_[1]-pos[1];
        return targetEnabled_ && pos[0]>=0 && pos[0]<=targetRadius_ && below>=targetDepth_ && below<=targetDepth_+targetThickness_;
    }'''+spatial[end:]
    (app/'spatialparams.hh').write_text(spatial)
    problem=(base/'evaporation/problem.hh').read_text()
    problem=replace_exact(problem,'pressure_ = 9.9e4;','pressure_ = getParam<Scalar>("Problem.InitialLiquidPressure", 96000.);')
    problem=replace_exact(problem,'temperatureInitial_ = 291;','temperatureInitial_ = getParam<Scalar>("Problem.InitialTemperature", 289.15);')
    start=problem.index('        BoundaryTypes values;')
    end=problem.index('        return values;',start)
    problem=problem[:start]+'''        BoundaryTypes values;
        values.setAllNeumann();
'''+problem[end:]
    start=problem.index('        NumEqVector values(0.0);')
    end=problem.index('        return values;',start)
    problem=problem[:start]+'''        NumEqVector values(0.0);
        const auto globalPos = scvf.ipGlobal();
        if (globalPos[dimWorld-1] > this->gridGeometry().bBoxMax()[dimWorld-1]-eps_)
        {
            const auto& volVars=elemVolVars[scvf.insideScvIdx()];
            const Scalar hour=std::fmod(getParam<Scalar>("Weather.StartHour",6.)+time_/3600.,24.);
            constexpr Scalar pi=3.141592653589793;
            const Scalar air=getParam<Scalar>("Weather.AirTemperature",289.15)+getParam<Scalar>("Weather.AirAmplitude",5.)*std::sin(2*pi*(hour-9)/24);
            const Scalar sky=air-getParam<Scalar>("Weather.SkyOffset",15.);
            const Scalar solar=getParam<Scalar>("Weather.SolarPeak",650.)*std::max(Scalar(0),std::sin(pi*(hour-6)/12));
            const Scalar wind=getParam<Scalar>("Weather.Wind",2.);
            const Scalar ts=volVars.temperature();
            // Vapour concentration difference across the unresolved boundary layer,
            // following the upstream ExtendedRichardsNI evaporation example.
            const Scalar inside=volVars.massFraction(FluidSystem::phase1Idx,FluidSystem::comp0Idx)*volVars.density(FluidSystem::phase1Idx);
            const Scalar tc=air-273.15;
            const Scalar ambient=getParam<Scalar>("Weather.RelativeHumidity",.55)*610.94*std::exp(17.625*tc/(tc+243.04))/(461.5*air);
            const Scalar diffusivity=volVars.effectiveDiffusionCoefficient(FluidSystem::phase1Idx,FluidSystem::comp1Idx,FluidSystem::comp0Idx);
            const Scalar thickness=getParam<Scalar>("Weather.BoundaryLayerThickness",.005)/(1+.5*wind);
            const Scalar evaporation=diffusivity*(inside-ambient)/thickness;
            values[conti0EqIdx]=evaporation;
            values[energyEqIdx]=FluidSystem::enthalpy(volVars.fluidState(),FluidSystem::gasPhaseIdx)*evaporation;
            const Scalar emissivity=getParam<Scalar>("Weather.Emissivity",.95);
            values[energyEqIdx]+=emissivity*5.670374419e-8*(std::pow(ts,4)-std::pow(sky,4))+(5.7+3.8*wind)*(ts-air)-(1-getParam<Scalar>("Weather.Albedo",.22))*solar;
        }
'''+problem[end:]
    problem=replace_exact(problem,'    const std::string& name() const','    void setTime(Scalar t) { time_=t; }\n\n    const std::string& name() const')
    problem=replace_exact(problem,'    Scalar temperatureInitial_;','    Scalar time_=0;\n    Scalar temperatureInitial_;')
    if args.surface:
        problem=replace_exact(problem,'#include <cmath>','#include <cmath>\n#include <array>\n#include <functional>')
        start=problem.index('        NumEqVector values(0.0);');end=problem.index('        return values;',start)
        problem=problem[:start]+'''        NumEqVector values(0.0);
        if (scvf.ipGlobal()[dimWorld-1] > this->gridGeometry().bBoxMax()[dimWorld-1]-eps_)
        {
            const auto& v=elemVolVars[scvf.insideScvIdx()];
            const Scalar distance=scvf.ipGlobal()[dimWorld-1]-element.geometry().center()[dimWorld-1];
            const auto state=surfaceState(v,distance);
            values[conti0EqIdx]=state[1];values[energyEqIdx]=state[2];
        }
'''+problem[end:]
        helper='''
    void setTimeFunction(std::function<Scalar()> function) { timeFunction_=std::move(function); }

    template<class VV>
    std::array<Scalar,3> surfaceState(const VV& v, Scalar distance) const
    {
        const Scalar at=timeFunction_ ? timeFunction_() : time_;
        const Scalar hour=std::fmod(getParam<Scalar>("Weather.StartHour",6.)+at/3600.,24.);
        constexpr Scalar pi=3.141592653589793;
        const Scalar air=getParam<Scalar>("Weather.AirTemperature",289.15)+getParam<Scalar>("Weather.AirAmplitude",5.)*std::sin(2*pi*(hour-9)/24);
        const Scalar sky=air-getParam<Scalar>("Weather.SkyOffset",15.);
        const Scalar solar=getParam<Scalar>("Weather.SolarPeak",650.)*std::max(Scalar(0),std::sin(pi*(hour-6)/12));
        const Scalar wind=getParam<Scalar>("Weather.Wind",2.);
        const Scalar ambient=getParam<Scalar>("Weather.RelativeHumidity",.55)*IapwsH2O::vaporPressure(air)/(461.5*air);
        const Scalar diffusivity=v.effectiveDiffusionCoefficient(FluidSystem::phase1Idx,FluidSystem::comp1Idx,FluidSystem::comp0Idx);
        const Scalar thickness=getParam<Scalar>("Weather.BoundaryLayerThickness",.005)/(1+.5*wind);
        const Scalar emissivity=getParam<Scalar>("Weather.Emissivity",.95);
        const Scalar resistance=distance/v.effectiveThermalConductivity();
        auto flux=[&](Scalar ts) {
            const Scalar activity=std::exp(-v.capillaryPressure()/(v.density(FluidSystem::phase0Idx)*461.5*ts));
            const Scalar inside=activity*IapwsH2O::vaporPressure(ts)/(461.5*ts);
            const Scalar evaporation=diffusivity*(inside-ambient)/thickness;
            auto state=v.fluidState();state.setTemperature(FluidSystem::gasPhaseIdx,ts);
            // Only water crosses this boundary: use water's component enthalpy,
            // not the enthalpy of the entire nitrogen/water gas mixture.
            const Scalar hv=FluidSystem::componentEnthalpy(state,FluidSystem::gasPhaseIdx,FluidSystem::comp0Idx);
            const Scalar potential=getParam<bool>("Problem.EnableGravity") ? 9.81*this->gridGeometry().bBoxMax()[dimWorld-1] : 0.;
            const Scalar energy=(hv+potential)*evaporation+emissivity*5.670374419e-8*(std::pow(ts,4)-std::pow(sky,4))+(5.7+3.8*wind)*(ts-air)-(1-getParam<Scalar>("Weather.Albedo",.22))*solar;
            return std::array<Scalar,2>{evaporation,energy};
        };
        Scalar ts=v.temperature();
        for (int i=0;i<20;++i)
        {
            const auto f=flux(ts);
            const Scalar residual=ts-v.temperature()+resistance*f[1];
            if (std::abs(residual)<1e-9) return {ts,f[0],f[1]};
            const Scalar derivative=1+resistance*(flux(ts+.001)[1]-flux(ts-.001)[1])/.002;
            ts-=residual/derivative;
            if (!(ts>250 && ts<370)) DUNE_THROW(Dune::InvalidStateException,"Surface temperature outside supported unfrozen domain");
        }
        DUNE_THROW(Dune::InvalidStateException,"Surface energy balance failed to converge");
    }
'''
        problem=replace_exact(problem,'    void setTime(Scalar t)',helper+'\n    void setTime(Scalar t)')
        problem=replace_exact(problem,'    Scalar time_=0;','    std::function<Scalar()> timeFunction_;\n    Scalar time_=0;')
    if args.cached_parameters and args.surface:
        begin=problem.index('    std::array<Scalar,3> surfaceState(')
        end=problem.index('    void setTime(Scalar t)',begin)
        block=problem[begin:end]
        constants={
            'Weather.StartHour':('startHour', '6.'),'Weather.AirTemperature':('airBase','289.15'),
            'Weather.AirAmplitude':('airAmplitude','5.'),'Weather.SkyOffset':('skyOffset','15.'),
            'Weather.SolarPeak':('solarPeak','650.'),'Weather.Wind':('windConstant','2.'),
            'Weather.RelativeHumidity':('humidity','.55'),'Weather.BoundaryLayerThickness':('boundaryLayer','.005'),
            'Weather.Emissivity':('emissivityConstant','.95'),'Weather.Albedo':('albedo','.22')}
        for key,(name,default) in constants.items():block=re.sub(r'getParam<Scalar>\("'+re.escape(key)+r'",\s*[^)]*\)',name,block)
        block=block.replace('getParam<bool>("Problem.EnableGravity")','gravityEnabled')
        declaration=''.join(f'        static const Scalar {name}=getParam<Scalar>("{key}",{default});\n' for key,(name,default) in constants.items())
        declaration+='        static const bool gravityEnabled=getParam<bool>("Problem.EnableGravity");\n'
        block=replace_exact(block,'        const Scalar at=',declaration+'        const Scalar at=')
        problem=problem[:begin]+block+problem[end:]
    (app/'problem.hh').write_text(problem)
    main=(base/'evaporation/main.cc').read_text()
    main=replace_exact(main,'        // solve the non-linear system with time step control','        problem->setTime(timeLoop->time());\n        // solve the non-linear system with time step control')
    if args.surface:
        main=replace_exact(main,'        problem->setTime(timeLoop->time());','        problem->setTimeFunction([timeLoop]() { return timeLoop->time()+timeLoop->timeStepSize(); });')
        main=replace_exact(main,'        timeLoop->advanceTimeStep();','        timeLoop->advanceTimeStep();\n        problem->setTimeFunction([timeLoop]() { return timeLoop->time(); });')
        insertion='''
    std::vector<Scalar> surfaceTemperatures(gridGeometry->numDofs());
    auto updateSurfaceFields=[&]() {
        auto geom=localView(*gridGeometry);
        auto vars=localView(gridVariables->curGridVolVars());
        for (const auto& element:elements(leafGridView))
        {
            geom.bind(element);vars.bind(element,geom,x);
            for (const auto& scv:scvs(geom)) surfaceTemperatures[scv.dofIndex()]=vars[scv].temperature();
            for (const auto& face:scvfs(geom))
                if (face.boundary() && face.ipGlobal()[GridGeometry::GridView::dimensionworld-1]>gridGeometry->bBoxMax()[GridGeometry::GridView::dimensionworld-1]-1e-9)
                {
                    constexpr int axis=GridGeometry::GridView::dimensionworld-1;
                    const auto& v=vars[face.insideScvIdx()];
                    const Scalar distance=face.ipGlobal()[axis]-element.geometry().center()[axis];
                    surfaceTemperatures[face.insideScvIdx()]=problem->surfaceState(v,distance)[0];
                }
        }
    };
    vtkWriter.addField(surfaceTemperatures,"surface_temperature");
    updateSurfaceFields();
'''
        main=replace_exact(main,'    vtkWriter.write(0.0);',insertion+'\n    vtkWriter.write(0.0);')
        main=replace_exact(main,'        vtkWriter.write(timeLoop->time());','        updateSurfaceFields();\n        vtkWriter.write(timeLoop->time());')
    if args.ledger:
        main=replace_exact(main,'#include <iostream>','#include <iostream>\n#include <fstream>\n#include <iomanip>\n#include <array>')
        main=replace_exact(main,'    // time loop\n', '''    // Direct diagnostics use the SAME storage and flux functions as DuMux's
    // finite-volume residual, including water vapour and fluid/solid energy.
    auto calculateBudget = [&]() {
        std::array<Scalar,10> b{};
        auto geom=localView(*gridGeometry);
        auto vars=localView(gridVariables->curGridVolVars());
        auto fluxCache=localView(gridVariables->gridFluxVarsCache());
        auto residual=assembler->localResidual();
        for (const auto& element : elements(leafGridView,Dune::Partitions::interior))
        {
            geom.bind(element);vars.bind(element,geom,x);fluxCache.bind(element,geom,vars);
            for (const auto& scv : scvs(geom))
            {
                const auto& v=vars[scv];
                const auto storage=residual.computeStorage(*problem,scv,v);
                const Scalar volume=scv.volume()*v.extrusionFactor();
                b[0]+=storage[0]*volume;b[1]+=storage[1]*volume;
                if (v.porosity()<1e-4) b[4]+=storage[0]*volume;
            }
            for (const auto& face : scvfs(geom))
            {
                if (face.boundary())
                {
                    const auto f=problem->neumann(element,geom,vars,fluxCache,face);
                    b[2]+=f[0]*face.area();b[3]+=f[1]*face.area();
                }
                else if ((vars[face.insideScvIdx()].porosity()<1e-4) != (vars[face.outsideScvIdx()].porosity()<1e-4))
                {
                    const auto f=residual.computeFlux(*problem,element,geom,vars,face,fluxCache);
                    if (vars[face.insideScvIdx()].porosity()<1e-4) b[5]+=std::abs(f[0]);
                    b[6]+=f[0];b[7]+=f[1];b[8]+=std::abs(f[0]);b[9]+=std::abs(f[1]);
                }
            }
        }
        for (auto& v:b) v=leafGridView.comm().sum(v);
        return b;
    };
    const auto initialBudget=calculateBudget();
    Scalar massOut=0,energyOut=0,absoluteMassOut=0,absoluteEnergyOut=0,targetExchange=0;
    std::ofstream budget;
    if (mpiHelper.rank()==0) {
        budget.open("conservation.csv");budget<<std::setprecision(17);
        budget<<"time_s,water_kg,energy_j,cumulative_water_out_kg,cumulative_energy_out_j,water_residual_kg,energy_residual_j,water_relative_residual,energy_relative_residual,target_water_kg,cumulative_absolute_target_exchange_kg,interface_water_net_kg_s,interface_energy_net_w,interface_water_absolute_kg_s,interface_energy_absolute_w\\n";
    }

    // time loop
''')
        main=replace_exact(main,'        // make the new solution the old solution','''        const auto b=calculateBudget();
        const Scalar diagnosticDt=timeLoop->timeStepSize();
        massOut+=b[2]*diagnosticDt;energyOut+=b[3]*diagnosticDt;
        absoluteMassOut+=std::abs(b[2])*diagnosticDt;absoluteEnergyOut+=std::abs(b[3])*diagnosticDt;
        targetExchange+=b[5]*diagnosticDt;
        const Scalar massResidual=b[0]-initialBudget[0]+massOut;
        const Scalar energyResidual=b[1]-initialBudget[1]+energyOut;
        const Scalar massScale=std::max({absoluteMassOut,std::abs(b[0]-initialBudget[0]),initialBudget[0]*1e-6});
        const Scalar energyScale=std::max({absoluteEnergyOut,std::abs(b[1]-initialBudget[1]),std::abs(initialBudget[1])*1e-9});
        if (mpiHelper.rank()==0) budget<<timeLoop->time()+diagnosticDt<<","<<b[0]<<","<<b[1]<<","<<massOut<<","<<energyOut<<","<<massResidual<<","<<energyResidual<<","<<std::abs(massResidual)/massScale<<","<<std::abs(energyResidual)/energyScale<<","<<b[4]<<","<<targetExchange<<","<<b[6]<<","<<b[7]<<","<<b[8]<<","<<b[9]<<"\\n";
        budget.flush();

        // make the new solution the old solution''')
    if args.compact_output:
        main=replace_exact(main,'#include <iostream>','#include <iostream>\n#include <algorithm>')
        main=replace_exact(main,'    std::vector<Scalar> surfaceTemperatures(gridGeometry->numDofs());','''    struct SurfacePoint { std::size_t index; Scalar x,y; };
    std::vector<SurfacePoint> surfacePoints;
    std::vector<Scalar> surfaceMoisture(gridGeometry->numDofs());
    std::vector<Scalar> surfaceTemperatures(gridGeometry->numDofs());''')
        main=replace_exact(main,'    auto updateSurfaceFields=[&]() {','    auto updateSurfaceFields=[&]() {\n        const bool collectCoordinates=surfacePoints.empty();')
        main=replace_exact(main,'                    surfaceTemperatures[face.insideScvIdx()]=problem->surfaceState(v,distance)[0];','''                    surfaceTemperatures[face.insideScvIdx()]=problem->surfaceState(v,distance)[0];
                    surfaceMoisture[face.insideScvIdx()]=v.porosity()*v.saturation(0);
                    if (collectCoordinates) surfacePoints.push_back({static_cast<std::size_t>(face.insideScvIdx()),face.ipGlobal()[0],face.ipGlobal()[1]});''')
        main=replace_exact(main,'    };\n    vtkWriter.addField(surfaceTemperatures,','''        if (collectCoordinates) std::sort(surfacePoints.begin(),surfacePoints.end(),[](const auto& a,const auto& b) { return a.y==b.y ? a.x<b.x : a.y<b.y; });
    };
    vtkWriter.addField(surfaceTemperatures,''')
        output='''
    if (leafGridView.comm().size()!=1) DUNE_THROW(Dune::NotImplemented,"Compact surface export currently requires sequential grid ownership");
    std::ofstream coordinates("surface-grid.csv");coordinates<<std::setprecision(17)<<"x_m,y_m\\n";
    for (const auto& p:surfacePoints) coordinates<<p.x<<","<<p.y<<"\\n";
    coordinates.close();
    const unsigned short byteTest=1;
    std::ofstream format("surface-format.json");
    format<<"{\\"points\\":"<<surfacePoints.size()<<",\\"scalar\\":\\"float64\\",\\"byte_order\\":\\""<<(*reinterpret_cast<const unsigned char*>(&byteTest)==1 ? "little" : "big")<<"\\",\\"layout\\":\\"time_s,temperature_k[N],moisture[N]\\"}";
    format.close();
    std::ofstream surfaceBinary("surface-frames.bin",std::ios::binary);
    auto writeSurfaceSnapshot=[&](Scalar at) {
        auto writeDouble=[&](Scalar value) { const double v=value;surfaceBinary.write(reinterpret_cast<const char*>(&v),sizeof(v)); };
        writeDouble(at);
        for (const auto& p:surfacePoints) writeDouble(surfaceTemperatures[p.index]);
        for (const auto& p:surfacePoints) writeDouble(surfaceMoisture[p.index]);
        surfaceBinary.flush();
    };
    writeSurfaceSnapshot(0.);
    Scalar lastVolumeOutput=0.;
    const Scalar volumeInterval=getParam<Scalar>("Output.VolumeInterval",3600.);
'''
        main=replace_exact(main,'    vtkWriter.write(0.0);',output+'\n    vtkWriter.write(0.0);')
        main=replace_exact(main,'        updateSurfaceFields();\n        vtkWriter.write(timeLoop->time());','''        updateSurfaceFields();
        writeSurfaceSnapshot(timeLoop->time());
        if (timeLoop->time()-lastVolumeOutput>=volumeInterval || timeLoop->finished())
        {
            vtkWriter.write(timeLoop->time());lastVolumeOutput=timeLoop->time();
        }''')
    if args.amgx:
        main=replace_exact(main,'#include <iostream>','#include <iostream>\n#include "amgx_solver.hh"')
        main=replace_exact(main,'        nonLinearSolver.solve(x, *timeLoop);','        try { nonLinearSolver.solve(x, *timeLoop); }\n        catch (const std::exception& error) { std::cerr << "GPU-capable Newton solve failed: " << error.what() << std::endl; return 2; }')
        main=replace_exact(main,'using LinearSolver = Dumux::ILUBiCGSTABIstlSolver<LinearSolverTraits<GridGeometry>, LinearAlgebraTraitsFromAssembler<Assembler>>;','using LinearSolver = AeroRozum::GpuOrIstlSolver<Dumux::ILUBiCGSTABIstlSolver<LinearSolverTraits<GridGeometry>, LinearAlgebraTraitsFromAssembler<Assembler>>>;')
        shutil.copy2(Path(__file__).with_name('amgx_solver.hh'),app/'amgx_solver.hh')
        shutil.copytree(Path(__file__).with_name('vendor'),app/'vendor',dirs_exist_ok=True)
    if args.axisymmetric:
        main=replace_exact(main,'const Scalar volume=scv.volume()*v.extrusionFactor();','const Scalar volume=Extrusion_t<GridGeometry>::volume(geom,scv)*v.extrusionFactor();')
        main=replace_exact(main,'b[2]+=f[0]*face.area();b[3]+=f[1]*face.area();','b[2]+=f[0]*Extrusion_t<GridGeometry>::area(geom,face);b[3]+=f[1]*Extrusion_t<GridGeometry>::area(geom,face);')
    if args.mpi_compact:
        if not args.compact_output:raise ValueError('MPI compact variant requires compact output')
        main=replace_exact(main,'#include <iostream>','#include <iostream>\n#include <numeric>\n#include <mpi.h>')
        main=replace_exact(main,'if (collectCoordinates) surfacePoints.push_back','if (collectCoordinates && (element.partitionType()==Dune::InteriorEntity || element.partitionType()==Dune::BorderEntity)) surfacePoints.push_back')
        first=main.index('    if (leafGridView.comm().size()!=1) DUNE_THROW')
        last=main.index('    writeSurfaceSnapshot(0.);',first)
        main=main[:first]+Path(__file__).with_name('mpi_surface_output.inc').read_text()+'\n'+main[last:]
    (app/'main.cc').write_text(main)
    (app/'CMakeLists.txt').write_text(f'dumux_add_test(NAME {binary_name} SOURCES main.cc COMPILE_DEFINITIONS TYPETAG=RichardsNIEvaporationCC)\n'+(f'target_link_libraries({binary_name} PRIVATE ${{CMAKE_DL_LIBS}})\n' if args.amgx else ''))
    cmake=base/'CMakeLists.txt'
    directive=f'add_subdirectory({app_name})'
    if directive not in cmake.read_text():cmake.write_text(cmake.read_text()+'\n'+directive+'\n')
    build=src/'build-reference'
    log=real/('campaign'+file_suffix+'-build.log')
    with log.open('a') as out:
        for command in [['cmake','-S',str(src),'-B',str(build)],['cmake','--build',str(build),'--target',binary_name,'-j',str(args.jobs)]]:
            subprocess.run(command,stdout=out,stderr=subprocess.STDOUT,check=True)
    binary=build/'test/porousmediumflow/richards/nonisothermal'/app_name/binary_name
    manifest=dict(status='custom_application_built',solver='DuMux',version='3.10.0',binary=str(binary),cached=args.cached,ledger=args.ledger,surface=args.surface,graded=args.graded,cached_parameters=args.cached_parameters,compact_output=args.compact_output,amgx_adapter=args.amgx,research_accepted=False,source_hashes={str(p.relative_to(app)):hashlib.sha256(p.read_bytes()).hexdigest() for p in app.rglob('*') if p.is_file()},limitations=['Experimental custom atmospheric boundary; no research acceptance or field validation.','Target near-impermeability is regularized and needs sensitivity tests.','One effective solid target; no resolved multilayer casing, rainfall or vegetation physiology.'])
    manifest['axisymmetric']=args.axisymmetric
    manifest['mpi_compact']=args.mpi_compact
    if args.axisymmetric:manifest['symmetry_scope']='Native RotationalExtrusion<0>; single vertical cylindrical inclusion, homogeneous flat soil and spatially uniform forcing only; separate candidate admission required'
    (real/('campaign'+file_suffix+'-status.json')).write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
