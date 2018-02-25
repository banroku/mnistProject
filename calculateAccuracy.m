function [correctRate correctList] = calculateAccuracy(Y, prediction);
    correctList = (Y == prediction);
    correctRate = sum(correctList)/size(correctList,2);

end

