n = 70;
total_fsim = 0;
for i=0:n-1
    generated_img = imread("exp13/" + i + ".jpg");
    truth_img = imread("truth-sketches/truth_sketch_" + i + ".jpg");
    [fsim, fsimc] = FeatureSIM(generated_img, truth_img);
    total_fsim = total_fsim + fsim;
end

fprintf("%g\n", total_fsim/n);



    