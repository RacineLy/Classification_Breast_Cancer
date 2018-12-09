% Function for plots
% ------------------------------------------------------------------------------
% Plot Learning Curves

%h1 = figure(1);
%plot((1:m), error_train, '-r','LineWidth',1.5); hold on;
%plot((1:m), error_valid, '-k','LineWidth',1.5);  
%xlabel('Training set size', 'FontSize',20, 'FontName', 'Times'); 
%ylabel('Error', 'FontSize',20, 'FontName', 'Times');  
%legend('Training Error','Validation Error', 'Location','SouthEast');
%legend boxoff;     
%saveas(h1,'Learning Curves','png');  

% Plot Validation Curves
h2 = figure(2);
plot((1:numel(lambdavec)), error_train_vc, '-r','LineWidth',1.5); hold on;
plot((1:numel(lambdavec)), error_valid_vc, '-k','LineWidth',1.5);  
xlabel('Regularization parameter', 'FontSize',20, 'FontName', 'Times'); 
ylabel('Error', 'FontSize',20, 'FontName', 'Times');  
legend('Training Error','Validation Error', 'Location','NorthWest');
legend boxoff;     
saveas(h2,'Validation Curves','png');
hold off;  

Corr_Xtrain = corr(Xtrain);
h3 = figure(3);
imagesc(Corr_Xtrain);
set(gca, 'XTick', 1:9, ...                             % Change the axes tick marks
         'XTickLabel', {}, ...  %   and tick labels
         'YTick', 1:9, ...
         'YTickLabel', {'Clump Thickness', 'Uniformity of Cell Size', ...
         'Uniformity of Cell Shape', 'Marginal Adhesion', ...
         'Single Epithelial Cell Size', 'Bare Nuclei',...
         'Bland Chromatin', 'Normal Nucleoli', 'Mitoses'}, ...
         'TickLength', [0 90]);
saveas(h3,'Training Dataset Correlation','png');


[U, V] = unique(ytrain);
h4 = figure(4);
labels = {'Benignes 33%','Malignes 67%'};
pie(U,V, labels);
saveas(h4,'Benign and Malignant stats','png');


h5 = figure(5);
plot(cost,'-*');
xlabel('Nombre d''iterations', 'FontSize',20, 'FontName', 'Times');
ylabel('Erreur', 'FontSize',20, 'FontName', 'Times');
saveas(h5,'Learning Cost','png'); 

