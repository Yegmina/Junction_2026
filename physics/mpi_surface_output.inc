    // SPDX-License-Identifier: GPL-3.0-or-later
    // Physics and linear algebra remain native DuMux. Only owned diagnostic
    // face values are gathered, preserving every solved physical time.
    const int surfaceRank=leafGridView.comm().rank(),surfaceRanks=leafGridView.comm().size();
    const auto surfaceComm=Dune::MPIHelper::getCommunicator();
    const int localSurfaceCount=static_cast<int>(surfacePoints.size());
    std::vector<int> surfaceCounts(surfaceRanks),surfaceOffsets(surfaceRanks);
    MPI_Gather(&localSurfaceCount,1,MPI_INT,surfaceCounts.data(),1,MPI_INT,0,surfaceComm);
    int globalSurfaceCount=0;
    if(surfaceRank==0) for(int p=0;p<surfaceRanks;++p) {surfaceOffsets[p]=globalSurfaceCount;globalSurfaceCount+=surfaceCounts[p];}
    std::vector<double> localSurfaceX(localSurfaceCount),localSurfaceY(localSurfaceCount),localSurfaceT(localSurfaceCount),localSurfaceM(localSurfaceCount);
    for(int i=0;i<localSurfaceCount;++i) {localSurfaceX[i]=surfacePoints[i].x;localSurfaceY[i]=surfacePoints[i].y;}
    std::vector<double> globalSurfaceX(globalSurfaceCount),globalSurfaceY(globalSurfaceCount),globalSurfaceT(globalSurfaceCount),globalSurfaceM(globalSurfaceCount);
    MPI_Gatherv(localSurfaceX.data(),localSurfaceCount,MPI_DOUBLE,globalSurfaceX.data(),surfaceCounts.data(),surfaceOffsets.data(),MPI_DOUBLE,0,surfaceComm);
    MPI_Gatherv(localSurfaceY.data(),localSurfaceCount,MPI_DOUBLE,globalSurfaceY.data(),surfaceCounts.data(),surfaceOffsets.data(),MPI_DOUBLE,0,surfaceComm);
    std::vector<int> surfaceOrder(globalSurfaceCount);std::iota(surfaceOrder.begin(),surfaceOrder.end(),0);
    std::sort(surfaceOrder.begin(),surfaceOrder.end(),[&](int a,int b) {return globalSurfaceY[a]==globalSurfaceY[b] ? globalSurfaceX[a]<globalSurfaceX[b] : globalSurfaceY[a]<globalSurfaceY[b];});
    std::ofstream surfaceBinary;
    if(surfaceRank==0) {
        std::ofstream coordinates("surface-grid.csv");coordinates<<std::setprecision(17)<<"x_m,y_m\n";
        for(const auto i:surfaceOrder) coordinates<<globalSurfaceX[i]<<","<<globalSurfaceY[i]<<"\n";
        const unsigned short byteTest=1;std::ofstream format("surface-format.json");
        format<<"{\"points\":"<<globalSurfaceCount<<",\"scalar\":\"float64\",\"byte_order\":\""<<(*reinterpret_cast<const unsigned char*>(&byteTest)==1 ? "little" : "big")<<"\",\"mpi_ranks\":"<<surfaceRanks<<",\"ownership\":\"interior_or_border\",\"layout\":\"time_s,temperature_k[N],moisture[N]\"}";
        surfaceBinary.open("surface-frames.bin",std::ios::binary);
    }
    auto writeSurfaceSnapshot=[&](Scalar at) {
        for(int i=0;i<localSurfaceCount;++i) {localSurfaceT[i]=surfaceTemperatures[surfacePoints[i].index];localSurfaceM[i]=surfaceMoisture[surfacePoints[i].index];}
        MPI_Gatherv(localSurfaceT.data(),localSurfaceCount,MPI_DOUBLE,globalSurfaceT.data(),surfaceCounts.data(),surfaceOffsets.data(),MPI_DOUBLE,0,surfaceComm);
        MPI_Gatherv(localSurfaceM.data(),localSurfaceCount,MPI_DOUBLE,globalSurfaceM.data(),surfaceCounts.data(),surfaceOffsets.data(),MPI_DOUBLE,0,surfaceComm);
        if(surfaceRank==0) {
            auto writeDouble=[&](double value) {surfaceBinary.write(reinterpret_cast<const char*>(&value),sizeof(value));};
            writeDouble(at);
            for(const auto i:surfaceOrder) writeDouble(globalSurfaceT[i]);
            for(const auto i:surfaceOrder) writeDouble(globalSurfaceM[i]);
            surfaceBinary.flush();
        }
    };
