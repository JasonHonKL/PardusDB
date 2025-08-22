#include <stdio.h>
#include <stdint.h>


enum KV_Type{
        // The value is a 8-bit unsigned integer.
    GGUF_METADATA_VALUE_TYPE_UINT8 = 0,
    // The value is a 8-bit signed integer.
    GGUF_METADATA_VALUE_TYPE_INT8 = 1,
    // The value is a 16-bit unsigned little-endian integer.
    GGUF_METADATA_VALUE_TYPE_UINT16 = 2,
    // The value is a 16-bit signed little-endian integer.
    GGUF_METADATA_VALUE_TYPE_INT16 = 3,
    // The value is a 32-bit unsigned little-endian integer.
    GGUF_METADATA_VALUE_TYPE_UINT32 = 4,
    // The value is a 32-bit signed little-endian integer.
    GGUF_METADATA_VALUE_TYPE_INT32 = 5,
    // The value is a 32-bit IEEE754 floating point number.
    GGUF_METADATA_VALUE_TYPE_FLOAT32 = 6,
    // The value is a boolean.
    // 1-byte value where 0 is false and 1 is true.
    // Anything else is invalid, and should be treated as either the model being invalid or the reader being buggy.
    GGUF_METADATA_VALUE_TYPE_BOOL = 7,
    // The value is a UTF-8 non-null-terminated string, with length prepended.
    GGUF_METADATA_VALUE_TYPE_STRING = 8,
    // The value is an array of other values, with the length and type prepended.
    ///
    // Arrays can be nested, and the length of the array is the number of elements in the array, not the number of bytes.
    GGUF_METADATA_VALUE_TYPE_ARRAY = 9,
    // The value is a 64-bit unsigned little-endian integer.
    GGUF_METADATA_VALUE_TYPE_UINT64 = 10,
    // The value is a 64-bit signed little-endian integer.
    GGUF_METADATA_VALUE_TYPE_INT64 = 11,
    // The value is a 64-bit IEEE754 floating point number.
    GGUF_METADATA_VALUE_TYPE_FLOAT64 = 12,
};

int main() {
    const char *filename = "model.gguf";
    FILE *f = fopen(filename, "rb");
    if (!f) {
        perror("Failed to open");
        return 1;
    }

    char magic[5] = {0}; // 4 chars + null terminator
    size_t bytes_read = fread(magic, 1, 4, f);
    if (bytes_read != 4) {
        printf("Failed to read magic\n");
        fclose(f);
        return 1;
    }
    printf("Magic: %s\n", magic);

    uint32_t version;
    bytes_read = fread(&version, sizeof(version), 1, f);
    if (bytes_read != 1) {
        printf("Failed to read version\n");
        fclose(f);
        return 1;
    }
    printf("Version: %u\n", version);

    uint64_t tensor_counter; 

    bytes_read = fread(&tensor_counter , sizeof(tensor_counter) , 1 , f);
    printf("Tensor count %llu\n" , tensor_counter);


    uint64_t metadata_kv_count;

    bytes_read = fread(&metadata_kv_count , sizeof(metadata_kv_count) , 1 , f);

    printf("Key value count %llu\n" , metadata_kv_count);

    //for(int i =0 ; i < metadata_kv_count; i++){

        uint64_t key_size;
        bytes_read = fread(&key_size , sizeof(key_size) , 1 , f);

        char key[key_size];
        bytes_read = fread(key , sizeof(char) ,key_size , f);

        printf("Key: %s\n", key); 

        uint32_t meta_data_value;
        bytes_read = fread(&meta_data_value , sizeof(meta_data_value) ,1 , f);
        printf("Type: %u\n" ,meta_data_value );

        if(meta_data_value == 8){
            // read the meta data
            uint64_t size_of_value;
            bytes_read = fread(&size_of_value , sizeof(size_of_value) , 1 ,f);

            printf("Size of value %llu\n" , size_of_value);
            
            char value[size_of_value];
            fread(&value , sizeof(char) , size_of_value , f);
            printf("Value: %s\n", value); 

        }
    //}

    fclose(f);
    return 0;
}