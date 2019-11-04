m(:,1) = [2;0]; Sigma(:,:,1) = 0.1*[5 0;0,2]; % mean and covariance of data pdf conditioned on label 2
m(:,2) = [0;-2]; Sigma(:,:,2) = 0.1*[7 1;1 2]; % mean and covariance of data pdf conditioned on label 1
classPriors = [0.3,0.7]; thr = [0,cumsum(classPriors)];
N = 999; u = rand(1,N); L = zeros(1,N); x = zeros(2,N);p = zeros(2,999);d = zeros(3,N);a=0;y = zeros(1,N)
figure(1),clf, colorList = 'rb';
Sb = (m(:,1)-m(:,2))'*(m(:,1)-m(:,2));
Sw = Sigma(:,:,1) + Sigma(:,:,2);
[V,D] = eig(inv(Sw)*Sb);
[~,ind] = sort(diag(D),'descend');
w = V(:,ind(1)); % Fisher LDA projection vector
smu = w'*m;
SSigma(1) = w'*Sigma(:,:,1)*w;
SSigma(2) = w'*Sigma(:,:,2)*w;
b = 0.8744;


for l = 1:2
    indices = find(thr(l)<=u & u<thr(l+1));
    L(1,indices) = l*ones(1,length(indices));
    x(:,indices) = mvnrnd(m(:,l),Sigma(:,:,l),length(indices))';
    figure(1), plot(x(1,indices),x(2,indices),'.','MarkerFaceColor',colorList(l)); axis equal, hold on,
    figure(1),legend('c1','c2')
    y(1,indices) = w'*x(:,indices);
end

for i  = 1:999
    xl = w'*x(:,i)+b;
    
    if (xl>0)
        d(1,i) = 1;
    else
        d(1,i) = 2;
    end
end

for i = 1:999
    pm1 = 0.3*evalGaussian(x(:,i),m(:,1),Sigma(:,:,1));
    pm2 = 0.7*evalGaussian(x(:,i),m(:,2),Sigma(:,:,2));
    if (pm1 - pm2 > 0)
        d(2,i) = 1;
    else
        d(2,i) = 2;
    end
end

we0 = [w(1),w(2),b];
[theta,fval] =  fminsearch(@(we)(fe(x,we,L)),we0);
for i = 1:999
    pe1 = 1/(1+exp(theta(2)*x(1,i)+theta(2)*x(2,i)+theta(3)));
    pe2 = 1-1/(1+exp(theta(2)*x(1,i)+theta(2)*x(2,i)+theta(3)));
    if (pe1 - pe2 > 0)
        d(3,i) = 1;
    else
        d(3,i) = 2;
    end
end
c1 = 0; c2 = 0; c3 = 0;

for i = 1:999
    if (d(1,i)==L(i))
        c1 = c1+1;
        if(L(i)==1)
            figure(2),scatter(x(1,i),x(2,i),'.g'),hold on,
        else
            figure(2),scatter(x(1,i),x(2,i),'+g'),hold on,
        end
    else
        if(L(i)==1)
            figure(2),scatter(x(1,i),x(2,i),'.r'),hold on,
        else
            figure(2),scatter(x(1,i),x(2,i),'+r'),hold on,
        end
    end
end
figure(2),
xlabel('x(1)'), ylabel('x(2)')
for i = 1:999
    if (d(2,i)==L(i))
        c2 = c2 +1;
        if(L(i)==1)
            figure(3),scatter(x(1,i),x(2,i),'.g'),hold on,
        else
            figure(3),scatter(x(1,i),x(2,i),'+g'),hold on,
        end
    else
        if(L(i)==1)
            figure(3),scatter(x(1,i),x(2,i),'.r'),hold on,
        else
            figure(3),scatter(x(1,i),x(2,i),'+r'),hold on,
        end
    end
end
figure(3),
xlabel('x(1)'), ylabel('x(2)')
for i = 1:999
    if (d(3,i)==L(i))
        c3 = c3+1;
        if(L(i)==1)
            figure(4),scatter(x(1,i),x(2,i),'.g'),hold on,
        else
            figure(4),scatter(x(1,i),x(2,i),'+g'),hold on,
        end
    else
        if(L(i)==1)
            figure(4),scatter(x(1,i),x(2,i),'.r'),hold on,
        else
            figure(4),scatter(x(1,i),x(2,i),'+r'),hold on,
        end
    end
end
figure(4),
xlabel('x(1)'), ylabel('x(2)')

disp('the MAP correct rate is :');
disp(c2/999);
disp('the LDA correct rate is :');
disp(c1/999);

disp('the LR correct rate is :');
disp(c3/999);
function g = evalGaussian(x,mu,Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
[n,N] = size(x);
invSigma = inv(Sigma);
C = (2*pi)^(-n/2) * det(invSigma)^(1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(invSigma*(x-repmat(mu,1,N))),1);
g = C*exp(E);
end
function f = fe(x,we,L)
m = length(L);
f = 0;
for i = 1:m
    if(L(i)==1)
        f =  f-log(1/(1+exp(we(2)*x(1,i)+we(2)*x(2,i)+we(3))));
    else
        f = f-log(1-1/(1+exp(we(2)*x(1,i)+we(2)*x(2,i)+we(3))));
    end
    
end
end
