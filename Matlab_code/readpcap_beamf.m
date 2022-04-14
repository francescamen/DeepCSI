classdef readpcap_beamf < handle
    
    properties
        fid;
    end
    
    methods
        function open(obj, filename, skip_start)
            obj.fid = fopen(filename);
            fread(obj.fid, skip_start, '*uint8');
        end
        
        function frame = next(obj, payload_length, skip_start_sample)
                fread(obj.fid, skip_start_sample, '*uint8');

                fread(obj.fid, 2, '*uint8');
                frame.header.header_length = fread(obj.fid, 1, '*uint8');

            try
                frame.header.radiotap_header = fread(obj.fid, frame.header.header_length - 3, '*uint8');

            catch
                disp('end of file')
                frame.payload = {};
                return
            end
                
            frame.header.control_field = fread(obj.fid, 2, '*uint8');
            frame.header.duration = fread(obj.fid, 2, '*uint8');

            frame.header.destination_mac = dec2hex(fread(obj.fid, 6, '*uint8'));
            frame.header.source_mac = dec2hex(fread(obj.fid, 6, '*uint8'));
            frame.header.bss_id = dec2hex(fread(obj.fid, 6, '*uint8'));
            frame.header.seq_frag_number = fread(obj.fid, 2, '*uint8');

            frame.header.category = fread(obj.fid, 1, '*uint8');
            fread(obj.fid, 1, '*uint8');
            frame.header.mimo_control = fread(obj.fid, 3, '*uint8');

            frame.payload = fread(obj.fid, payload_length, '*uint8');
            frame.FCS = fread(obj.fid, 4, '*uint8');

        end
        
    end
    
end

