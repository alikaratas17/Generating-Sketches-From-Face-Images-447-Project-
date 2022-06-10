n = 1;
total_fsim = 0;
for i=1:n
    generated_img = imread("generated_sketch_" + i + ".jpg");
    truth_img = imread("true_sketch_" + i + ".jpg");
    [fsim, fsimc] = FeatureSIM(generated_img, truth_img);
    total_fsim = total_fsim + fsim;
end

fprintf("%g", total_fsim/n);



    