#include "hashmap.h"

uint8_t hash_i(uint64_t i, uint32_t capacity){
	return i % capacity;
}

uint8_t hash_s(const char* key, uint32_t capacity){
	uint32_t hash = 5381;
	int16_t c;
	while ((c=*key++)) hash = ((hash<<5)+hash)+c;
	return c%capacity;
}
