edf = 'F:\wsc_edf\A0001_4 165907.EDf';
hdr = loadHDR(edf);
channels = {'C3-x','O1-x','LEOG-x','ROC-M1','Chin EMG'};
fs = 100;
sig = load_signal(edf, fs);

fs = 100;

%Filtering variables
fcH = 0.2/(fs/2); %relative frequency/2

[bH,aH] = butter(5,fcH,'high');

fcL = num2cell([49 25 12.5 6.25 3.125]/(fs/2));

[bL,aL] = cellfun(@(x) butter(5,x,'low'), fcL,'Un',0);

segLength = 120*10;
overlap = 40*10;

sig = cellfun(@(x) filtfilt(bH,aH,x),sig,'Un',0);
sig = cellfun(@(x) single(filtfilt(bL{1},aL{1},x)),sig,'Un',0);
C = cell(6,1);
dim = [2 2 4 4 0.4 4];
for j=1:numel(sig)    
    C{j} = extractCC(fs,dim(j),0.25,sig{j}',sig{j}');
end
C{6} = extractCC(fs,dim(6),0.25,sig{3}',sig{4}');