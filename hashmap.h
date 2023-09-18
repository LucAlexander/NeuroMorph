#ifndef TSHM_H
#define TSHM_H

#include <stddef.h>
#include <inttypes.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

uint8_t hash_i(uint64_t key, uint32_t capacity);
uint8_t hash_s(const char* key, uint32_t capacity);

#define HASHMAP_CAPACITY 31

#define HASHMAP(typename, keyType, valType) \
	typedef struct typename##TSHM_NODE{ \
		struct typename##TSHM_NODE* next; \
		keyType key; \
		valType val; \
	}typename##TSHM_NODE; \
	\
	typedef struct typename{ \
		typename##TSHM_NODE** data; \
		uint32_t capacity; \
		uint32_t size; \
	}typename; \
	\
	typedef struct typename##_result{ \
		uint8_t error; \
		keyType key; \
		valType val; \
	}typename##_result; \
	\
	typedef struct typename##_iterator{ \
		typename* map; \
		typename##TSHM_NODE* current; \
		int32_t index; \
	}typename##_iterator; \
	\
	typename typename##_init(); \
	\
	typename##TSHM_NODE* typename##TSHM_NODE_INIT(keyType key, valType val); \
	\
	typename##_iterator typename##_iterator_init(typename* map); \
	\
	typename##_result typename##_iterator_next(typename##_iterator* it); \
	\
	uint8_t typename##_iterator_has_next(typename##_iterator* it); \
	\
	void typename##TSHM_NODE_FREE(typename##TSHM_NODE* n); \
	\
	void typename##_free(typename* map); \
	\
	uint8_t typename##TSHM_NODE_INSERT(typename##TSHM_NODE* n, keyType key, valType val); \
	\
	keyType* typename##_get_key_set(typename* map); \
	\
	void typename##_push(typename* map, keyType key, valType val); \
	\
	typename##_result typename##_get(typename* map, keyType key); \
	\
	valType* typename##_ref(typename* map, keyType key); \
	\
	uint8_t typename##_contains(typename* map, keyType key); \
	\
	typename##_result typename##_pop(typename* map, keyType key); \
	\
	void typename##_clear(typename* map); \
	\


#define HASHMAP_SOURCE(typename, keyType, valType, hashing) \
	typename typename##_init(){ \
		uint32_t cap = HASHMAP_CAPACITY; \
		typename##TSHM_NODE** d = calloc(cap, sizeof(typename##TSHM_NODE*)); \
		typename map = {d, cap, 0}; \
		return map; \
	} \
	\
	typename##TSHM_NODE* typename##TSHM_NODE_INIT(keyType key, valType val){ \
		typename##TSHM_NODE* n = malloc(sizeof(typename##TSHM_NODE)); \
		n->key = key; \
		n->val = val; \
		n->next = NULL; \
		return n; \
	} \
	\
	typename##_iterator typename##_iterator_init(typename* map){ \
		uint32_t i; \
		for (i = 0;i<map->capacity&&map->data[i]==NULL;++i){} \
		if (i==map->capacity){ \
			typename##TSHM_NODE* current = NULL; \
			typename##_iterator it = {map, current, -1}; \
			return it; \
		} \
		typename##TSHM_NODE* current = map->data[i]; \
		typename##_iterator it = {map, current, i}; \
		return it; \
	} \
	\
	typename##_result typename##_iteratorNext(typename##_iterator* it){ \
		typename##TSHM_NODE* temp = it->current; \
		if (!temp||it->index==-1){ \
			typename##_result res = {.error = 1}; \
			return res; \
		} \
		it->current = it->current->next; \
		if (it->current!=NULL){ \
			typename##_result res = {0, temp->key, temp->val}; \
			return res; \
		} \
		uint32_t i; \
		for (i = ++it->index;i<it->map->capacity&&it->map->data[i]==NULL;++i){} \
		if (i==it->map->capacity){ \
			it->index = -1; \
			typename##_result res = {0, temp->key, temp->val}; \
			return res; \
		} \
		it->current = it->map->data[i]; \
		it->index = i; \
		typename##_result res = {0, temp->key, temp->val}; \
		return res; \
	} \
	\
	uint8_t typename##_iterator_has_next(typename##_iterator* it){ \
		return (it!=NULL) && (it->current!=NULL) && (it->index != -1); \
	} \
	\
	void typename##TSHM_NODE_FREE(typename##TSHM_NODE* n){ \
		while(n){ \
			typename##TSHM_NODE* temp = n; \
			n = n->next; \
			free(temp); \
			temp = NULL; \
		} \
		n = NULL; \
	} \
	\
	void typename##_free(typename* map){ \
		uint32_t i; \
		map->size = 0; \
		for (i = 0;i<map->capacity;++i){ \
			if (map->data[i]){ \
				typename##TSHM_NODE_FREE(map->data[i]); \
			} \
		} \
		free(map->data); \
		map->data = NULL; \
	} \
	\
	uint8_t typename##TSHM_NODE_INSERT(typename##TSHM_NODE* n, keyType key, valType val){ \
		typename##TSHM_NODE* last = n; \
		while (n){ \
			if (n->key==key){ \
				n->val = val; \
				return 1; \
			} \
			last = n; \
			n = n->next; \
		} \
		last->next = typename##TSHM_NODE_INIT(key, val); \
		return 0; \
	} \
	\
	keyType* typename##_get_key_set(typename* map){ \
		typename##_iterator it = typename##_iterator_init(map); \
		keyType* set = malloc(map->size*sizeof(keyType)); \
		uint32_t i = 0; \
		while (typename##_iterator_has_next(&it)){ \
			set[i++] = typename##_iterator_next(&it).key; \
		} \
		return set; \
	} \
	\
	void typename##_push(typename* map, keyType key, valType val){ \
		uint32_t index = hashing(key, map->capacity); \
		if (map->data[index]){ \
			uint8_t replace = typename##TSHM_NODE_INSERT(map->data[index], key, val); \
			if (replace == 0){ \
				map->size++; \
			} \
			return; \
		} \
		map->size++; \
		map->data[index] = typename##TSHM_NODE_INIT(key, val); \
	} \
	\
	typename##_result typename##_get(typename* map, keyType key){ \
		if (!map || map->capacity==0){ \
			typename##_result res = {.error=2}; \
			return res; \
		} \
		uint32_t index = hashing(key, map->capacity); \
		typename##TSHM_NODE* temp; \
		temp = map->data[index]; \
		while(temp!=NULL){ \
			if (temp->key == key){ \
				typename##_result res = {0, temp->key, temp->val}; \
				return res; \
			} \
			temp = temp->next; \
		} \
		typename##_result res = {.error = 1}; \
		return res; \
	} \
	\
	valType* typename##_ref(typename* map, keyType key){ \
		if (!map || map->capacity==0){ \
			return NULL; \
		} \
		uint32_t index = hashing(key, map->capacity); \
		typename##TSHM_NODE* temp; \
		temp = map->data[index]; \
		while(temp!=NULL){ \
			if (temp->key == key){ \
				return &(temp->val); \
			} \
			temp = temp->next; \
		} \
		return NULL; \
	} \
	\
	uint8_t typename##_contains(typename* map, keyType key){ \
		return typename##_ref(map, key) != NULL; \
	} \
	\
	typename##_result typename##_pop(typename* map, keyType key){ \
		if (!map||map->size == 0){ \
			typename##_result res = {.error=2}; \
			return res; \
		} \
		uint32_t index = hashing(key, map->capacity); \
		typename##TSHM_NODE* temp = map->data[index]; \
		typename##TSHM_NODE* last = NULL; \
		while (temp){ \
			if (temp->key == key){ \
				if (!last){ \
					map->data[index] = temp->next; \
					temp->next = NULL; \
					keyType ky = temp->key; \
					valType vl = temp->val; \
					typename##TSHM_NODE_FREE(temp); \
					typename##_result res = {0,ky,vl}; \
					map->size--; \
					return res; \
				} \
				last->next = temp->next; \
				temp->next = NULL; \
				valType vl = temp->val; \
				keyType ky = temp->key; \
				typename##TSHM_NODE_FREE(temp); \
				typename##_result res = {0, ky, vl}; \
				map->size--; \
				return res; \
			} \
			last = temp; \
			temp = temp->next; \
		} \
		typename##_result res = {.error=1}; \
		return res; \
	} \
	\
	void typename##_clear(typename* map){ \
		uint32_t i; \
		for (i = 0;i<map->capacity;++i){ \
			typename##TSHM_NODE_FREE(map->data[i]); \
			map->data[i] = NULL; \
		} \
		map->size = 0; \
	} \
	\

#endif

