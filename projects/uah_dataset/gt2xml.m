% entrada: gt + folder + file_name + carpeta_destino
function gt2xml(gt_file_name, image_folder, xml_folder)
% gt2xml('./video240/video240.gt', './video240/frames/', './video240/xml/')

if image_folder(end) ~= '\' && image_folder(end) ~= '/'
    image_folder = [image_folder '/'];
end

if xml_folder(end) ~= '\' && xml_folder(end) ~= '/'
    xml_folder = [xml_folder '/'];
end

if ~exist(xml_folder, 'dir'); mkdir(xml_folder); end

gt_m = read_gt_file(gt_file_name); %read gt file
for i = 1:length(gt_m)
    % read info from gt
    if isempty(gt_m{i}); continue; end
    line = gt_m{i};
    nIm = line(1); npeople = (length(line)-2)/6;

    % read image file name
    d = dir(sprintf('%s/*%.4d.jpg', image_folder, nIm));
    if isempty(d); continue; end
    [p n e] = fileparts(d(1).name);
    im = imread(sprintf('%s%s', image_folder, d(1).name));
    [rows cols ch] = size(im);

    xml_file = fullfile(xml_folder,[n '.xml']);
    fid = fopen(xml_file, 'w');

    write_xml_header(fid, image_folder, n, e, cols, rows, ch);

    for i = 1:npeople
        data = line(3*(2*i-1):3*(2*i)+2);
        add_person(fid, data) ;
    end

     fprintf(fid, '</annotation>\n');
     fclose(fid);
end

function write_xml_header(fid, image_folder, n, e, cols, rows, ch);
fprintf(fid, '<annotation>\n');
fprintf(fid, '\t<folder>%s</folder>\n', image_folder);
fprintf(fid, '\t<filename>%s%s</filename>\n', n, e);
fprintf(fid, '\t<path>%s%s%s</path>\n', image_folder, n, e);
fprintf(fid, '\t<source>\n\t\t<database>Unknown</database>\n\t</source>\n');
fprintf(fid, '\t<size>\n\t\t<width>%d</width>\n', cols);
fprintf(fid, '\t\t<height>%d</height>\n', rows);
fprintf(fid, '\t\t<depth>%d</depth>\n\t</size>\n', ch);
fprintf(fid, '\t<segmented>0</segmented>\n');



function add_person(fid, data)
id = data(1);
fprintf(fid, '\t<object>\n');
fprintf(fid, '\t\t<id>%d</id>\n', id);
fprintf(fid, '\t\t<pose>Unspecified</pose>\n');
fprintf(fid, '\t\t<truncated>0</truncated>\n');
fprintf(fid, '\t\t<difficult>0</difficult>\n');
fprintf(fid, '\t\t<bndbox>\n');
fprintf(fid, '\t\t\t<xmin>%d</xmin>\n', round(data(2)));
fprintf(fid, '\t\t\t<ymin>%d</ymin>\n', round(data(3)));
fprintf(fid, '\t\t\t<xmax>%d</xmax>\n', round(data(4)));
fprintf(fid, '\t\t\t<ymax>%d</ymax>\n', round(data(5)));
fprintf(fid, '\t\t</bndbox>\n');
fprintf(fid, '\t</object>\n');


%----
function gt_m = read_gt_file(gt_file)

gt_m = {};
fid = fopen(gt_file,'r');
aux = fgets(fid);
fin = 0;
if aux ~= -1
    while ~fin
        aux2 = str2num(aux);
        if ~isempty(aux2(1))
            if aux2(1) == 0; a = 1; elseif aux2(1) > 0; a = 0; end
            fin = 1; break;
        else
            aux = fgets(fid);
        end
    end
end

while aux ~= -1
    aux2 = str2num(aux);
    if ~isempty(aux2)
        % test if .gt includes, or not, action information
        if length(aux2) == 7
            aux2 = [aux2 0];
        elseif ~mod(length(aux2)-2, 5)
            if floor(aux2(8)) ~= aux2(8) || floor(aux2(9)) ~= aux(9)
                aux3 = [aux2(1:7) 0];
                for j = 8:5:length(aux2)
                    aux3 = [aux3 aux2(j:j+4) 0];
                end
                aux2 = aux3;
            end
        end
        gt_m{aux2(1) + a} = aux2; %str2num(aux);
    end
    aux = fgets(fid);
end
fclose(fid);

