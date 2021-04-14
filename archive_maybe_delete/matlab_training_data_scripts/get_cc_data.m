%This script generates the cross-correlation data used for the convolutional neural networks.

%Initialize variables

fs = 100;

%Filtering variables
fcH = 0.2/(fs/2); %relative frequency/2

[bH,aH] = butter(5,fcH,'high');

fcL = num2cell([49 25 12.5 6.25 3.125]/(fs/2));

[bL,aL] = cellfun(@(x) butter(5,x,'low'), fcL,'Un',0);

segLength = 120*10;
overlap = 40*10;
addpath('/home/jenss/Documents/MATLAB/13_10_15/sev')
import filter.*
import plist.*;

%In case labels are not in the same folder as EDFs
%directory for saving data
saveDir = '';
%paths where the EDF files can be found
paths = {''};

%Make list of files.
listF = [];
for i=1:length(paths)
    listT = dir(paths{i});
    listT = {listT(:).name};
    listT = listT(3:end);
    
    index = strfind(listT,'edf');
    index = not(cellfun('isempty',index));
    
    index2 = strfind(listT,'EDF');
    index2 = not(cellfun('isempty',index2));
    
    index = or(index,index2);
    
    listT = strcat(paths{i},listT);
    
    listT = listT(index);
    listF = [listF listT];
end

rng(12345)
listF = listF(randperm(length(listF)));

channels = {'C3M2','O1M2','LEOGM2','REOGM1','Chin1Chin2','EKG'};
try
    parpool(4)
end

i = 0;
while true
   i = i+1;
    try
    if i<length(listF)
    disp(num2str(i))
    dataName = listF{i}(1:end-4);
    disp(dataName)
    ind = strfind(dataName,'/');
    name = dataName(ind(end)+1:end);
    dataName = ['data_' name];
    %find matching hypnogram
  	hyp = load_scored_dat([listF{i}(1:end-4) '.STA']);
	if isempty(hyp)
		disp('Skipping due to missing hypnogram')
		continue
	end

	sig = load_signal(listF{i},fs);
    if isempty(sig)
		disp('Skipping due to missing edf')
		continue
	end

        hyp = hyp(1:length(sig{1})/(fs*30));
        label = repmat(hyp',120,1);
        label = label(:);
        
        sig = cellfun(@(x) filtfilt(bH,aH,x),sig,'Un',0);
        sig = cellfun(@(x) single(filtfilt(bL{1},aL{1},x)),sig,'Un',0);
        C = cell(6,1);
	dim = [2 2 4 4 0.4 4];
        parfor j=1:5%length(sig)   
            
            C{j} = extractCC(fs,dim(j),0.25,sig{j}',sig{j}');
        end
        C{6} = extractCC(fs,dim(6),0.25,sig{3}',sig{4}');
        
	disp('Data extracted')

        C = cellfun(@(x) x(:,1:size(C{6},2)),C,'Un',0);
        C = vertcat(C{1},C{2},C{3},C{4},C{6},C{5});
        label = label(1:size(C,2));
        C(:,label==7) = [];
        label(label==7) = [];
        hyp(hyp==7) = [];
        
        labels = zeros(5,length(label));
        
        
        for j=1:5
            labels(j,label==j) = 1;
        end
        
        index = num2cell(buffer(1:size(C,2),segLength,overlap,'nodelay'),1);
        index(end) = [];

        M = cellfun(@(x) C(:,x),index,'Un',0);
        M = cat(3,M{:});

        L = cellfun(@(x) labels(:,x),index,'Un',0);
        L = cat(3,L{:});

        hyp = repmat(hyp,1,2)';
        hyp = hyp(:);
        
        if ~exist('dataStack','var')
            dataStack = (M);
            labelStack = (L);
            weightStack = (W);
        else
            dataStack = cat(3,dataStack,M);
            labelStack = cat(3,labelStack,L);
            weightStack = cat(2,weightStack,W);
        end

    end
    catch
        warning([listF{i} ' caused an error'])
    end
    if size(dataStack,3) > 900 || (i==length(listF) && size(dataStack,3) > 300)
        
        if ~exist(saveDir,'dir')
            mkdir(saveDir)
        end
        here = cd
        cd(saveDir)
        
        
        rng('shuffle')
        
        ind = randperm(size(dataStack,3));
        dataStack = dataStack(:,:,ind);
        labelStack = labelStack(:,:,ind);
        weightStack = weightStack(:,ind);
        saveName = [num2str(randi([10000000 19999999])) '.h5'];
        
        h5create(saveName,'/trainD',[1640 segLength 270]);
        h5write(saveName, '/trainD', dataStack(:,:,1:270));
        
        h5create(saveName,'/trainL',[5 segLength 270]);
        h5write(saveName, '/trainL', labelStack(:,:,1:270));
        
        h5create(saveName,'/keep',[segLength 270]);
        h5write(saveName, '/keep', ones(segLength,270));
        
        h5create(saveName,'/valD',[1640 segLength 30]);
        h5write(saveName, '/valD', dataStack(:,:,271:300));
        
        h5create(saveName,'/valL',[5 segLength 30]);
        h5write(saveName, '/valL', labelStack(:,:,271:300));
        
        
        dataStack(:,:,1:300) = [];
        labelStack(:,:,1:300) = [];
        weightStack(:,1:300) = [];

        cd(here)
    elseif i==length(listF)
        break
        
    end
end 

