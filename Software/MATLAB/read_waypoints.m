%% ============================================================
% GENERAR waypoints.json DESDE ROBOTAT
% =============================================================
clc;
robotat = robotat_connect();

% ------------------------------------------------------------
% CONFIGURACIÓN
% ------------------------------------------------------------
marker_ids   = [61 63 64 65 67];
HEIGHT_START = 0.50;   % offset para el primer marker (inicio)
HEIGHT_END   = 1;   % offset para el último marker (fin)
HEIGHT_MID   = -0.12;   % offset para los markers intermedios (obstáculos)
output_file  = 'waypoints.json';

DRONE_ID     = 82;     % ID del dron en Robotat -- solo para el chequeo

% ------------------------------------------------------------
% DIAGNÓSTICO -- posición actual del dron (referencia, no se guarda)
% ------------------------------------------------------------
% Objetivo: confirmar si robotat_get_pose() ve al dron con una posición
% real, sin pasar por el puente MQTT que usa flight_purepursuit.py
fprintf('--- Diagnóstico: posición del dron (ID %d) vía Robotat directo ---\n', DRONE_ID);
try
    pose_dron = robotat_get_pose(robotat, DRONE_ID, 'eulxyz');
    fprintf('Dron (ID %d) -> x=%.3f  y=%.3f  z=%.3f\n\n', ...
            DRONE_ID, pose_dron(1), pose_dron(2), pose_dron(3));
catch err
    fprintf('⚠ No se pudo leer la posición del dron (ID %d): %s\n\n', ...
            DRONE_ID, err.message);
end

% ------------------------------------------------------------
% LEER MARKERS
% ------------------------------------------------------------
n          = length(marker_ids);
waypoints  = zeros(n, 3);
offsets    = [HEIGHT_START, repmat(HEIGHT_MID, 1, n-2), HEIGHT_END];

fprintf('Leyendo markers...\n\n');
for i = 1:n
    pose = robotat_get_pose(robotat, marker_ids(i), 'eulxyz');
    x = pose(1);
    y = pose(2);
    z = pose(3);
    fprintf('Marker %d -> x=%.3f  y=%.3f  z=%.3f  (offset=%.2f)\n', ...
            marker_ids(i), x, y, z, offsets(i));
    waypoints(i,:) = [x, y, z + offsets(i)];
end

% ------------------------------------------------------------
% GENERAR JSON
% ------------------------------------------------------------
jsonText = jsonencode(waypoints);
fid = fopen(output_file, 'w');
if fid == -1
    error('No se pudo crear waypoints.json');
end
fprintf(fid, '%s', jsonText);
fclose(fid);
fprintf('\nArchivo "%s" generado correctamente.\n\n', output_file);

% ------------------------------------------------------------
% MOSTRAR WAYPOINTS
% ------------------------------------------------------------
disp('Waypoints generados:')
for i = 1:n
    label = '';
    if i == 1,  label = '  ← inicio'; end
    if i == n,  label = '  ← fin';    end
    fprintf('WP%d: x=%.3f  y=%.3f  z=%.3f%s\n', ...
        i-1, waypoints(i,1), waypoints(i,2), waypoints(i,3), label);
end
