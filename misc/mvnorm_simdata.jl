using Distributions

covmat = [1.0 0.6 0.2 0.0;
          0.6 1.0 0.6 0.2;
          0.2 0.6 1.0 0.6;
          0.0 0.2 0.6 1.0]

D = MvNormal([1.0, 2.0, -3.0, 3.0], covmat)

X = rand(D, 10000)'
cor(X)
