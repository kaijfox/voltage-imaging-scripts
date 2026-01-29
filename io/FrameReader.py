# Translation of FrameReader.m to python
#
# NOTE: The following functionality from the original MATLAB code has been OMITTED:
# 1. All TIFF-related methods (readTiff, readTiffs, memory mapping, etc.)
# 2. TIFF folder setup and metadata extraction
# 3. CSV/MAT/XML metadata loading (getTm method)
# 4. Image description parsing for TIFF files
# 5. Memory mapping capabilities
# 6. BigTIFF support and specialized TIFF readers
#
# Python-specific considerations:
# 1. Used struct module for binary file reading instead of MATLAB's fread
# 2. Used numpy arrays instead of MATLAB matrices
# 3. Used pathlib and glob for file operations
# 4. Error handling adapted to Python exceptions
# 5. Properties implemented using @property decorator

import os
import re
import numpy as np
import glob
from pathlib import Path
from typing import Optional, Tuple, Union, List
import struct


class FrameReader:
    """
    Python translation of FrameReader.m for RAW file functionality only.

    TIFF functionality has been omitted - only RAW file reading is supported.
    """

    def __init__(self, file_id: str, w: int = 588, h: int = 624, g: int = 1728,
                 dtype: str = 'uint8', max_frames: int = 24000, shuffle: bool = False):
        """
        Initialize FrameReader for RAW files.

        Args:
            file_id: Path to RAW file or folder containing RAW files
            w: Frame width (default 588)
            h: Frame height (default 624)
            g: Gap between frames in bytes (default 1728)
            dtype: Data type ('uint8' or 'uint16', default 'uint8')
            max_frames: Maximum number of frames (default 24000)
            shuffle: Whether to shuffle frames (default False)
        """
        self.file_id = file_id
        self.w = w
        self.h = h
        self.g = g
        self.first_offset = 0
        self.dtype = dtype
        self.max_frames = max_frames
        self.shuffle = shuffle
        self.pointer = 1 - 2 * self.shuffle  # -1 if shuffle else 1
        self.is_tiff = False  # Always False for this RAW-only version
        self.is_folder = False
        self.raw_file_names = []
        self.metadata = None
        self.region = None

        # Initialize based on file type
        if os.path.isdir(self.file_id):
            self.is_folder = True
            self._setup_folder()
        else:
            # Single RAW file
            folder = os.path.dirname(self.file_id)
            self._find_and_parse_import_file(folder)
            self.max_frames = self._calculate_num_frames(self.file_id)

    @property
    def frame_size(self) -> int:
        """Calculate frame size in pixels."""
        return self.w * self.h

    @property
    def info(self) -> dict:
        """Return dictionary of key FrameReader properties."""
        return {
            'file_id': self.file_id,
            'is_folder': self.is_folder,
            'width': self.w,
            'height': self.h,
            'frame_size': self.frame_size,
            'gap_bytes': self.g,
            'dtype': self.dtype,
            'first_offset': self.first_offset,
            'max_frames': self.max_frames,
            'current_pointer': self.pointer,
            'shuffle_mode': self.shuffle,
            'has_region': self.region is not None,
            'region': self.region,
            'num_raw_files': len(self.raw_file_names) if self.is_folder else (1 if not self.is_folder else 0),
            'raw_file_names': self.raw_file_names if self.is_folder else None,
            'has_metadata': self.metadata is not None
        }

    def _setup_folder(self):
        """Setup for folder containing multiple RAW files."""
        # NOTE: TIFF folder functionality omitted

        # Look for RAW files
        raw_pattern = os.path.join(self.file_id, '*.raw')
        raw_files = glob.glob(raw_pattern)

        if not raw_files:
            raise ValueError('No RAW files found in directory')

        self.raw_file_names = [os.path.basename(f) for f in raw_files]

        # Extract frame numbers from filenames
        frame_nums = []
        for filename in self.raw_file_names:
            match = re.search(r'(\d+)\.raw$', filename)
            if match:
                frame_nums.append(int(match.group(1)))

        if not frame_nums:
            raise ValueError('Could not extract frame numbers from RAW filenames')

        print(f"[FrameReader] Finding and parsing RAW files in folder: {self.file_id}")
        self._find_and_parse_import_file(self.file_id)

        # Calculate total frames across all files
        self.max_frames = 0
        for raw_file in self.raw_file_names:
            full_path = os.path.join(self.file_id, raw_file)
            self.max_frames += self._calculate_num_frames(full_path)

    def _find_and_parse_import_file(self, folder: str):
        """Find and parse import*.txt file to get frame parameters."""
        import_files = glob.glob(os.path.join(folder, '*import*.txt'))

        if len(import_files) == 1:
            self.w, self.h, self.g, self.first_offset, self.dtype = self._parse_file_info(import_files[0])

        # Handle ROI for files with firstOffset=80
        if self.first_offset == 80:
            if self.is_folder and self.raw_file_names:
                roi_file = os.path.join(folder, self.raw_file_names[0])
            else:
                roi_file = self.file_id
            self.region = self._get_roi_raw(roi_file)

    def _parse_file_info(self, file_info_path: str) -> Tuple[int, int, int, int, str]:
        """Parse import file to extract frame parameters."""
        with open(file_info_path, 'r') as f:
            content = f.read()

        # Regular expressions for extracting parameters
        width_match = re.search(r'- Width: (\d+)', content)
        height_match = re.search(r'- Height: (\d+)', content)
        gap_match = re.search(r'- Gap between images: (\d+)', content)
        offset_match = re.search(r'- Offset to first image: (\d+)', content)
        image_type_match = re.search(r"- Image Type: '(\d+)-bit\s*\w*'", content)

        width = int(width_match.group(1)) if width_match else self.w
        height = int(height_match.group(1)) if height_match else self.h
        gap = int(gap_match.group(1)) if gap_match else self.g
        first_offset = int(offset_match.group(1)) if offset_match else self.first_offset

        if image_type_match:
            bits = int(image_type_match.group(1))
            dtype = f'uint{bits}'
        else:
            dtype = self.dtype

        return width, height, gap, first_offset, dtype

    def _get_roi_raw(self, filename: str) -> List[int]:
        """Extract ROI information from first 80 bytes of RAW file."""
        try:
            with open(filename, 'rb') as f:
                # Read first 80 bytes as uint16 (little-endian)
                data = struct.unpack('<40H', f.read(80))  # 40 uint16 values

            # Extract region coordinates (1-indexed as in MATLAB)
            region = [data[29] + 1, data[30] + 1, data[32] + 1, data[33] + 1]
            return region
        except Exception as e:
            print(f"Error reading ROI from {filename}: {e}")
            return None

    def _get_pixel_bytes(self, dtype: str) -> int:
        """Get number of bytes per pixel for given data type."""
        if dtype == 'uint8':
            return 1
        elif dtype == 'uint16':
            return 2
        else:
            raise ValueError(f"Unsupported data type: {dtype}")

    def _calculate_num_frames(self, filename: str) -> int:
        """Calculate number of frames in a RAW file."""
        pixel_bytes = self._get_pixel_bytes(self.dtype)

        # Get file size
        file_size = os.path.getsize(filename)

        # Calculate number of frames
        n_frames = (file_size - self.first_offset) / ((self.frame_size * pixel_bytes) + self.g)

        # Check if frame count is close to an integer
        decimal_part = abs(n_frames - round(n_frames))
        if decimal_part > 1/50:
            if not getattr(self, 'warned_float_frames', False):
                print(f'Warning: Frame count is not within 1/50 of an integer: {n_frames}')
                self.warned_float_frames = True
            n_frames = int(np.floor(n_frames))

        return round(n_frames)

    def get_frames(self, num_frames: int, give_partial: bool = True) -> Optional[np.ndarray]:
        """
        Get next batch of frames from the file(s).

        Args:
            num_frames: Number of frames to read
            give_partial: Whether to return partial batches at end of file

        Returns:
            3D numpy array of shape (height, width, num_frames) or None if done
        """
        if self.is_done(num_frames, give_partial):
            return None

        num_frames = min(num_frames, self.max_frames - self.pointer + 1)
        start_frame = self._get_start_frame(num_frames)

        if self.is_folder:
            frames = self._read_raws(num_frames, start_frame)
        else:
            frames = self._read_raw(num_frames, start_frame)

        self.pointer += num_frames
        return frames

    def is_done(self, num_frames: int, give_partial: bool = True) -> bool:
        """Check if reading is complete."""
        return (self.pointer == self.max_frames + 1 or
                (not give_partial and self.pointer + num_frames > self.max_frames))

    def get_frames_by_indexes(self, start_idx: int, end_idx: int) -> np.ndarray:
        """
        Get frames by specific index range.

        Args:
            start_idx: Starting frame index (1-based)
            end_idx: Ending frame index (1-based, inclusive)

        Returns:
            3D numpy array of shape (height, width, num_frames)
        """
        assert end_idx <= self.max_frames, f'Invalid frame index: {end_idx}'
        assert start_idx >= 1, f'Invalid frame index: {start_idx}'
        assert start_idx == round(start_idx) and end_idx == round(end_idx), 'Both indices must be integers'

        num_frames = end_idx - start_idx + 1

        if self.is_folder:
            return self._read_raws(num_frames, start_idx)
        else:
            return self._read_raw(num_frames, start_idx)

    def _get_start_frame(self, num_frames: int) -> int:
        """Get starting frame for reading."""
        if self.pointer == -1:  # Random shuffle mode
            return np.random.randint(0, self.max_frames - num_frames + 1)
        else:
            return self.pointer

    def _read_raws(self, num_frames: int, start_frame: int) -> np.ndarray:
        """Read frames from multiple RAW files in a folder."""
        current_frame_index = 0
        frames = np.zeros((self.h, self.w, num_frames), dtype=self.dtype)

        while current_frame_index < num_frames:
            file_index, local_frame_index = self._get_raw_frame_location(current_frame_index + start_frame)
            filename = os.path.join(self.file_id, self.raw_file_names[file_index])
            frames_to_read = min(num_frames - current_frame_index,
                               self._calculate_num_frames(filename) - local_frame_index + 1)

            batch_frames = self._read_raw(frames_to_read, local_frame_index, filename)
            frames[:, :, current_frame_index:current_frame_index + frames_to_read] = batch_frames
            current_frame_index += frames_to_read

        return frames

    def _get_raw_frame_location(self, start_frame: int) -> Tuple[int, int]:
        """Map global frame index to specific file and local frame index."""
        cumulative_frames = 0
        file_index = 0
        local_frame_index = 0

        for i, raw_file in enumerate(self.raw_file_names):
            full_path = os.path.join(self.file_id, raw_file)
            num_frames_in_file = self._calculate_num_frames(full_path)

            if cumulative_frames + num_frames_in_file > start_frame:
                file_index = i
                local_frame_index = start_frame - cumulative_frames
                break

            cumulative_frames += num_frames_in_file

        return file_index, local_frame_index

    def _read_raw(self, num_frames: int, start_frame: int, filename: Optional[str] = None) -> np.ndarray:
        """
        Read frames from a single RAW file with batching for memory efficiency.

        Args:
            num_frames: Number of frames to read
            start_frame: Starting frame index (1-based, converted to 0-based internally)
            filename: Optional filename override

        Returns:
            3D numpy array of shape (height, width, num_frames)
        """
        if filename is None:
            filename = self.file_id


        pixel_bytes = self._get_pixel_bytes(self.dtype)
        batch_size = 1000
        num_batches = int(np.ceil(num_frames / batch_size))


        # Convert numpy dtype string to struct format
        dtype_map = {'uint8': 'B', 'uint16': 'H'}
        struct_format = dtype_map[self.dtype]

        batch_frames = []

        with open(filename, 'rb') as fid:
            file_size = fid.seek(0, 2)  # Get file size
            fid.seek(0)

            for b in range(num_batches):

                # Calculate frames in this batch
                if b == num_batches - 1:
                    frames_in_batch = num_frames - b * batch_size
                else:
                    frames_in_batch = batch_size

                # Calculate offset for this batch (convert to 0-based indexing)
                batch_start_frame = start_frame - 1 + b * batch_size
                offset = self.first_offset + (self.frame_size * pixel_bytes + self.g) * batch_start_frame
                fid.seek(offset)


                # Read data for this batch including gaps
                total_elements = frames_in_batch * self.frame_size
                total_gap = (frames_in_batch - 1) * self.g
                total_bytes = total_elements * pixel_bytes + total_gap


                data = fid.read(total_bytes)

                # Unpack binary data
                format_str = f'<{len(data)//pixel_bytes}{struct_format}'
                unpacked_data = struct.unpack(format_str, data[:len(data)//pixel_bytes*pixel_bytes])
                unpacked_data = np.array(unpacked_data, dtype=self.dtype)


                # Handle gap removal following MATLAB logic
                if self.g > 0:
                    gap_pixels = self.g // pixel_bytes

                    # Extend data to match MATLAB's padding approach
                    unpacked_data = np.append(unpacked_data, np.ones(gap_pixels, dtype=self.dtype))

                    # Reshape to separate frames as columns (each column = frame + gap)
                    pixels_per_frame_with_gap = self.frame_size + gap_pixels

                    unpacked_data = unpacked_data.reshape(pixels_per_frame_with_gap, frames_in_batch, order='F')

                    # Remove gap pixels from end of each frame column
                    unpacked_data = unpacked_data[:-gap_pixels, :]

                    # Reshape to proper frame dimensions
                    unpacked_data = unpacked_data.reshape(self.w, self.h, frames_in_batch, order='F')
                else:
                    unpacked_data = unpacked_data.reshape(self.w, self.h, frames_in_batch, order='F')

                batch_frames.append(unpacked_data)

        # Concatenate all batches
        frames = np.concatenate(batch_frames, axis=2)

        # Transpose to match MATLAB's dimension order (height, width, frames)
        frames = np.transpose(frames, (1, 0, 2))

        return frames


