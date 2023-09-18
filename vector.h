#ifndef TSV_H
#define TSV_H

#include <inttypes.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#define VECTOR(typename, type) \
	typedef struct typename{ \
		type* data; \
		uint32_t capacity; \
		uint32_t size; \
	}typename; \
 \
	typedef struct typename##_iterator{ \
		typename* vec; \
		uint32_t index; \
	}typename##_iterator; \
 \
	typename typename##_init(); \
 \
	typename##_iterator typename##_iterator_init(typename* vec); \
 \
	uint8_t typename##_iterator_has_next(typename##_iterator* it); \
 \
	type typename##_iterator_next(typename##_iterator* it); \
 \
	void typename##_free(typename* vec); \
 \
	void typename##_resize(typename* vec); \
 \
	void typename##_push(typename* vec, type item); \
 \
 	void typename##_insert(typename* vec, uint32_t index, type item); \
 \
	void typename##_reserve(typename* vec, uint32_t places); \
 \
 	void typename##_set(typename* vec, uint32_t index, type item); \
 \
	type typename##_remove(typename* vec, uint32_t index); \
 \
	type typename##_pop(typename* vec); \
 \
	void typename##_clear(typename* vec);  \
 \

#define VECTOR_SOURCE(typename, type) \
	typename typename##_init(){ \
		uint32_t cap = 32; \
		uint32_t s = 0; \
		type* data = malloc(cap*sizeof(type)); \
		typename vec = {data, cap, s}; \
		return vec; \
	} \
 \
	typename##_iterator typename##_iterator_init(typename* vec){ \
		typename##_iterator it = {vec, 0}; \
		return it; \
	} \
 \
	uint8_t typename##_iterator_has_next(typename##_iterator* it){ \
		return it->index < it->vec->size; \
	} \
 \
	type typename##_iterator_next(typename##_iterator* it){ \
		return it->vec->data[it->index++]; \
	} \
 \
	void typename##_free(typename* vec){ \
		free(vec->data); \
		vec->data = NULL; \
		vec->size = 0; \
	} \
 \
	void typename##_resize(typename* vec){ \
		type* temp = realloc(vec->data, vec->capacity*sizeof(type)*2); \
		vec->data = temp; \
		vec->capacity*=2; \
	} \
 \
	void typename##_push(typename* vec, type item){ \
		if (vec->size >= vec->capacity){ \
			typename##_resize(vec); \
		} \
		vec->data[vec->size++] = item; \
	} \
 \
	void typename##_insert(typename* vec, uint32_t index, type item){ \
		if (index >= vec->size){ \
			typename##_push(vec, item); \
			return; \
		} \
		if (vec->size >= vec->capacity){ \
			typename##_resize(vec); \
		} \
		vec->size++; \
		uint32_t i; \
		for (i = vec->size;i>0;--i){ \
			if (index == i){ \
				vec->data[i] = item; \
				return; \
			} \
			vec->data[i] = vec->data[i-1]; \
		} \
		vec->data[0] = item; \
	} \
 \
	void typename##_reserve(typename* vec, uint32_t places){ \
		if (vec->capacity-vec->size < places){ \
			uint32_t remaining = places - (vec->capacity-vec->size); \
			type* temp = realloc(vec->data, (vec->capacity+remaining)*sizeof(type)); \
			vec->data = temp; \
			vec->capacity += remaining; \
		} \
	} \
\
	void typename##_set(typename* vec, uint32_t index, type item){ \
		vec->data[index] = item; \
	} \
 \
	type typename##_remove(typename* vec, uint32_t index){ \
		type res = vec->data[index]; \
		vec->data[index] = vec->data[--vec->size]; \
		return res; \
	} \
 \
	type typename##_pop(typename* vec){ \
		return vec->data[--vec->size]; \
	} \
 \
	void typename##_clear(typename* vec){  \
		vec->size = 0; \
	} \
 \

#endif

