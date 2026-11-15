// SPDX-License-Identifier: GPL-3.0-or-later
// Optional in-process GPU linear solver for the unchanged DuMux Newton system.
#pragma once
#include <dlfcn.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>
#include "vendor/amgx_c.h"

namespace AeroRozum {
class AmgxApi {
    void* library_ = nullptr;
public:
#define AERO_API(name) decltype(&name) name##_ = nullptr;
    AERO_API(AMGX_initialize) AERO_API(AMGX_initialize_plugins)
    AERO_API(AMGX_finalize) AERO_API(AMGX_finalize_plugins)
    AERO_API(AMGX_config_create) AERO_API(AMGX_config_destroy)
    AERO_API(AMGX_resources_create_simple) AERO_API(AMGX_resources_destroy)
    AERO_API(AMGX_matrix_create) AERO_API(AMGX_matrix_destroy)
    AERO_API(AMGX_matrix_upload_all) AERO_API(AMGX_matrix_replace_coefficients)
    AERO_API(AMGX_vector_create) AERO_API(AMGX_vector_destroy)
    AERO_API(AMGX_vector_upload) AERO_API(AMGX_vector_download)
    AERO_API(AMGX_vector_set_zero) AERO_API(AMGX_vector_bind)
    AERO_API(AMGX_solver_create) AERO_API(AMGX_solver_destroy)
    AERO_API(AMGX_solver_setup) AERO_API(AMGX_solver_resetup)
    AERO_API(AMGX_solver_solve) AERO_API(AMGX_solver_get_status)
    AERO_API(AMGX_solver_get_iterations_number) AERO_API(AMGX_get_error_string)
#undef AERO_API
    explicit AmgxApi(const std::string& path) {
        library_=dlopen(path.c_str(),RTLD_NOW|RTLD_GLOBAL);
        if (!library_) throw std::runtime_error(std::string("AMGX dlopen failed: ")+dlerror());
#define AERO_LOAD(name) name##_ = reinterpret_cast<decltype(&name)>(dlsym(library_,#name)); if (!name##_) throw std::runtime_error("Missing AMGX symbol " #name);
        AERO_LOAD(AMGX_initialize) AERO_LOAD(AMGX_initialize_plugins)
        AERO_LOAD(AMGX_finalize) AERO_LOAD(AMGX_finalize_plugins)
        AERO_LOAD(AMGX_config_create) AERO_LOAD(AMGX_config_destroy)
        AERO_LOAD(AMGX_resources_create_simple) AERO_LOAD(AMGX_resources_destroy)
        AERO_LOAD(AMGX_matrix_create) AERO_LOAD(AMGX_matrix_destroy)
        AERO_LOAD(AMGX_matrix_upload_all) AERO_LOAD(AMGX_matrix_replace_coefficients)
        AERO_LOAD(AMGX_vector_create) AERO_LOAD(AMGX_vector_destroy)
        AERO_LOAD(AMGX_vector_upload) AERO_LOAD(AMGX_vector_download)
        AERO_LOAD(AMGX_vector_set_zero) AERO_LOAD(AMGX_vector_bind)
        AERO_LOAD(AMGX_solver_create) AERO_LOAD(AMGX_solver_destroy)
        AERO_LOAD(AMGX_solver_setup) AERO_LOAD(AMGX_solver_resetup)
        AERO_LOAD(AMGX_solver_solve) AERO_LOAD(AMGX_solver_get_status)
        AERO_LOAD(AMGX_solver_get_iterations_number) AERO_LOAD(AMGX_get_error_string)
#undef AERO_LOAD
        check(AMGX_initialize_());check(AMGX_initialize_plugins_());
    }
    void check(AMGX_RC result) const {
        if (result!=AMGX_RC_OK) { char message[4096]={};AMGX_get_error_string_(result,message,sizeof(message));throw std::runtime_error(std::string("AMGX: ")+message); }
    }
    ~AmgxApi() { if (library_) { AMGX_finalize_plugins_();AMGX_finalize_();dlclose(library_); } }
};

template<class Base> class GpuOrIstlSolver : public Base {
    std::unique_ptr<AmgxApi> api_;
    AMGX_config_handle config_{};
    AMGX_resources_handle resources_{};
    AMGX_matrix_handle matrix_{};
    AMGX_vector_handle rhs_{},solution_{};
    AMGX_solver_handle solver_{};
    bool uploaded_=false,exported_=false;
    std::vector<int> rowptr_,columns_;
    std::size_t calls_=0;
    double reduction_=1e-10;
    void initialize_() {
        if (api_) return;
        api_=std::make_unique<AmgxApi>(Dumux::getParam<std::string>("AMGX.LibraryPath","libamgxsh.so"));
        const std::string options=Dumux::getParam<std::string>("AMGX.Configuration",R"({"config_version":2,"solver":{"solver":"FGMRES","max_iters":1000,"gmres_n_restart":50,"tolerance":1e-12,"norm":"L2","convergence":"RELATIVE_INI_CORE","monitor_residual":1,"print_solve_stats":0,"preconditioner":{"solver":"BLOCK_JACOBI","max_iters":1,"relaxation_factor":0.9}}})");
        api_->check(api_->AMGX_config_create_(&config_,options.c_str()));
        api_->check(api_->AMGX_resources_create_simple_(&resources_,config_));
        api_->check(api_->AMGX_matrix_create_(&matrix_,resources_,AMGX_mode_dDDI));
        api_->check(api_->AMGX_vector_create_(&rhs_,resources_,AMGX_mode_dDDI));
        api_->check(api_->AMGX_vector_create_(&solution_,resources_,AMGX_mode_dDDI));
        api_->check(api_->AMGX_solver_create_(&solver_,resources_,AMGX_mode_dDDI,config_));
    }
public:
    using Base::Base;
    void setResidualReduction(double value) { reduction_=value;Base::setResidualReduction(value); }
    ~GpuOrIstlSolver() {
        if (api_) {
            if(solver_) api_->AMGX_solver_destroy_(solver_);
            if(solution_) api_->AMGX_vector_destroy_(solution_);
            if(rhs_) api_->AMGX_vector_destroy_(rhs_);
            if(matrix_) api_->AMGX_matrix_destroy_(matrix_);
            if(resources_) api_->AMGX_resources_destroy_(resources_);
            if(config_) api_->AMGX_config_destroy_(config_);
        }
    }
    template<class Matrix,class Vector> bool solve(Matrix& A,Vector& x,Vector& b) {
        using Clock=std::chrono::steady_clock;
        const auto began=Clock::now();
        const bool gpu=Dumux::getParam<bool>("AMGX.Enabled",false);
        const bool exportMatrix=Dumux::getParam<bool>("AMGX.ExportMatrix",false)&&!exported_;
        if (!gpu&&!exportMatrix) return Base::solve(A,x,b);
        const int n=static_cast<int>(A.N());
        constexpr int block=2;
        std::vector<int> rows(n+1,0),columns;
        std::vector<double> values,original,rhs(2*n),originalRhs(2*n),answer(2*n);
        for (auto row=A.begin();row!=A.end();++row) {
            const int i=row.index();
            double d00=0,d01=0,d10=0,d11=0;
            for(auto col=row->begin();col!=row->end();++col) if(col.index()==i) { d00=(*col)[0][0];d01=(*col)[0][1];d10=(*col)[1][0];d11=(*col)[1][1]; }
            const double determinant=d00*d11-d01*d10;
            if (!std::isfinite(determinant)||std::abs(determinant)<1e-300) throw std::runtime_error("Singular local 2x2 Jacobian block; GPU scaling rejected");
            const double p00=d11/determinant,p01=-d01/determinant,p10=-d10/determinant,p11=d00/determinant;
            originalRhs[2*i]=b[i][0];originalRhs[2*i+1]=b[i][1];
            rhs[2*i]=p00*b[i][0]+p01*b[i][1];rhs[2*i+1]=p10*b[i][0]+p11*b[i][1];
            for(auto col=row->begin();col!=row->end();++col) {
                columns.push_back(col.index());
                const double a=(*col)[0][0],c=(*col)[0][1],d=(*col)[1][0],e=(*col)[1][1];
                original.insert(original.end(),{a,c,d,e});
                values.insert(values.end(),{p00*a+p01*d,p00*c+p01*e,p10*a+p11*d,p10*c+p11*e});
            }
            rows[i+1]=columns.size();
        }
        if(exportMatrix) {
            auto writeMatrix=[&](const char* name,const std::vector<double>& data) {
                std::ofstream f(name);f<<"%%MatrixMarket matrix coordinate real general\n"<<2*n<<' '<<2*n<<' '<<4*columns.size()<<'\n'<<std::setprecision(17);
                for(int i=0;i<n;++i) for(int k=rows[i];k<rows[i+1];++k) for(int p=0;p<2;++p) for(int q=0;q<2;++q) f<<2*i+p+1<<' '<<2*columns[k]+q+1<<' '<<data[4*k+2*p+q]<<'\n';
            };
            auto writeVector=[&](const char* name,const std::vector<double>& data) {std::ofstream f(name);f<<"%%MatrixMarket matrix array real general\n"<<2*n<<" 1\n"<<std::setprecision(17);for(double v:data)f<<v<<'\n';};
            writeMatrix("jacobian-original.mtx",original);writeMatrix("jacobian-scaled.mtx",values);
            writeVector("rhs-original.mtx",originalRhs);writeVector("rhs-scaled.mtx",rhs);exported_=true;
        }
        if(!gpu) {
            const auto cpuStart=Clock::now();
            const bool good=Base::solve(A,x,b);
            const double elapsed=std::chrono::duration<double>(Clock::now()-cpuStart).count();
            if(exportMatrix) {
                std::ofstream f("correction-native-istl.mtx");f<<"%%MatrixMarket matrix array real general\n"<<2*n<<" 1\n"<<std::setprecision(17);
                for(int i=0;i<n;++i)for(int p=0;p<2;++p)f<<x[i][p]<<'\n';
                std::ofstream receipt("native-linear-baseline.json");receipt<<std::setprecision(17)<<"{\"backend\":\"ILUBiCGSTABIstlSolver\",\"solve_s\":"<<elapsed<<",\"requested_reduction\":"<<reduction_<<",\"passed\":"<<(good?"true":"false")<<"}\n";
            }
            return good;
        }
        initialize_();
        const auto prepared=Clock::now();
        const int nnz=columns.size();
        if(!uploaded_||rows!=rowptr_||columns!=columns_) {
            api_->check(api_->AMGX_matrix_upload_all_(matrix_,n,nnz,block,block,rows.data(),columns.data(),values.data(),nullptr));
            api_->check(api_->AMGX_vector_bind_(rhs_,matrix_));api_->check(api_->AMGX_vector_bind_(solution_,matrix_));
            rowptr_=rows;columns_=columns;uploaded_=true;
        } else api_->check(api_->AMGX_matrix_replace_coefficients_(matrix_,n,nnz,values.data(),nullptr));
        api_->check(api_->AMGX_solver_setup_(solver_,matrix_));
        api_->check(api_->AMGX_vector_upload_(rhs_,n,block,rhs.data()));api_->check(api_->AMGX_vector_set_zero_(solution_,n,block));
        const auto solveStart=Clock::now();
        api_->check(api_->AMGX_solver_solve_(solver_,rhs_,solution_));
        AMGX_SOLVE_STATUS status;int iterations=0;
        api_->check(api_->AMGX_solver_get_status_(solver_,&status));api_->check(api_->AMGX_solver_get_iterations_number_(solver_,&iterations));
        api_->check(api_->AMGX_vector_download_(solution_,answer.data()));
        const auto solved=Clock::now();
        double scaledResidual=0,scaledNorm=0,rawResidual=0,rawNorm=0;
        double componentResidual[2]={0,0},componentNorm[2]={0,0};
        for(int i=0;i<n;++i) for(int p=0;p<2;++p) {
            double r=-rhs[2*i+p],raw=-originalRhs[2*i+p];
            for(int k=rows[i];k<rows[i+1];++k) for(int q=0;q<2;++q) {r+=values[4*k+2*p+q]*answer[2*columns[k]+q];raw+=original[4*k+2*p+q]*answer[2*columns[k]+q];}
            scaledResidual+=r*r;scaledNorm+=rhs[2*i+p]*rhs[2*i+p];rawResidual+=raw*raw;rawNorm+=originalRhs[2*i+p]*originalRhs[2*i+p];
            componentResidual[p]+=raw*raw;componentNorm[p]+=originalRhs[2*i+p]*originalRhs[2*i+p];
            x[i][p]=answer[2*i+p];
        }
        const double scaledRelative=std::sqrt(scaledResidual/std::max(scaledNorm,1e-300)),rawRelative=std::sqrt(rawResidual/std::max(rawNorm,1e-300));
        const double waterRelative=std::sqrt(componentResidual[0]/std::max(componentNorm[0],1e-300)),energyRelative=std::sqrt(componentResidual[1]/std::max(componentNorm[1],1e-300));
        const bool good=status==AMGX_SOLVE_SUCCESS&&std::isfinite(scaledRelative)&&scaledRelative<=std::min(1e-8,reduction_)&&rawRelative<=std::min(1e-8,reduction_)&&waterRelative<=std::min(1e-8,reduction_)&&energyRelative<=std::min(1e-8,reduction_);
        std::ofstream log("gpu-linear-solves.jsonl",std::ios::app);
        log<<std::setprecision(17)<<"{\"call\":"<<++calls_<<",\"backend\":\"AMGX_fp64_BSR2_in_process\",\"cells\":"<<n<<",\"block_nnz\":"<<nnz<<",\"iterations\":"<<iterations<<",\"scaled_relative_residual\":"<<scaledRelative<<",\"original_relative_residual\":"<<rawRelative<<",\"water_relative_residual\":"<<waterRelative<<",\"energy_relative_residual\":"<<energyRelative<<",\"preparation_s\":"<<std::chrono::duration<double>(prepared-began).count()<<",\"setup_upload_s\":"<<std::chrono::duration<double>(solveStart-prepared).count()<<",\"solve_download_s\":"<<std::chrono::duration<double>(solved-solveStart).count()<<",\"passed\":"<<(good?"true":"false")<<"}\n";
        log.flush();
        if(!good) throw std::runtime_error("AMGX convergence/residual gate failed; no CPU fallback");
        return true;
    }
};
}
