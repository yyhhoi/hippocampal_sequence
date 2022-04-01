function xytsp = loadspikes( spikeData, ntrial, nu )

    xytsp.x=spikeData.xSp{ntrial,nu};
    xytsp.y=spikeData.ySp{ntrial,nu};
    xytsp.t=spikeData.tSp{ntrial,nu};

end

