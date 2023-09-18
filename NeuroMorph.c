#include <Python.h>
#include <stddef.h>
#include <stdlib.h>

#include <stdio.h>

#include "NeuroMorph.h"

VECTOR_SOURCE(vector, uintptr_t)
HASHMAP_SOURCE(adjacency_map, uintptr_t, vector, hash_i)

neuromorph_node* neuromorph_input_init(size_t input_size){
	neuromorph_node* node = neuromorph_divergent_init();
	node->buffer_size = input_size;
	node->neuron_buffer = malloc(sizeof(float)*node->buffer_size);
	return node;
}

neuromorph_node* neuromorph_divergent_init(){
	neuromorph_node* node = malloc(sizeof(neuromorph_node));
	node->next = NULL;
	node->prev = NULL;
	node->type = DIVERGENT_NODE;
	node->ready = 0;
	node->neuron_buffer = NULL;
	node->buffer_size = 0;
	node->weight_buffer = NULL;
	node->weight_buffer_size = 0;
	node->bias_buffer = NULL;
	node->bias_buffer_size = 0;
	node->activation_function = NULL;
	node->loss_function = NULL;
	node->previous_neuron_buffer = NULL;
	node->previous_buffer_size = NULL;
	node->additional_branches = NULL;
	node->additional_branch_count = 0;
	node->convergent_node = NULL;
	node->convergent_buffer = NULL;
	node->convergent_buffer_size = NULL;
	node->convergence_function = NULL;
	return node;
}

neuromorph_node* neuromorph_convergent_init(void (*convergence)(const float* const, float* const, const size_t* const)){
	neuromorph_node* node = neuromorph_divergent_init();
	node->convergence_function = convergence;
	return node;
}

neuromorph_node* neuromorph_layer_init(size_t buffer_size, void (*activation)(float* const, const size_t* const)){
	neuromorph_node* node = neuromorph_input_init(buffer_size);
	node->bias_buffer = malloc(sizeof(float)*buffer_size);
	node->bias_buffer_size = buffer_size;
	node->activation_function = activation;
	return node;
}

neuromorph_node* neuromorph_output_init(size_t buffer_size, void (*activation)(float* const, const size_t* const), void (*loss)(float* const, const size_t* const)){
	neuromorph_node* node = neuromorph_layer_init(buffer_size, activation);
	node->loss_function = loss;
	return node;
}

void neuromorph_node_free(neuromorph_node* node){
	free(node->neuron_buffer);
	free(node->weight_buffer);
	free(node->bias_buffer);
	free(node->additional_branches);
	free(node);
}

void neuromorph_link(adjacency_map* adjacency, neuromorph_node* const source, neuromorph_node* const destination){
	if (!(neuromorph_link_source(source, destination) && neuromorph_link_destination(source, destination))){
		return;
	}
	vector* v = adjacency_map_ref(adjacency,(uintptr_t)source);
	if (v){
		vector new_vector = vector_init();
		vector_push(&new_vector, (uintptr_t)destination);
		adjacency_map_push(adjacency, (uintptr_t)source, new_vector);
		return;
	}
	vector_push(v,(uintptr_t)destination);
}

uint8_t neuromorph_link_source(neuromorph_node* const source, neuromorph_node* const destination){
	switch(source->type){
	case INPUT_NODE:
	case CONVERGENT_NODE:
	case LAYER_NODE:
		source->next = destination;
		return 1;
	case DIVERGENT_NODE:
		if (source->next != NULL){
			if (source->additional_branch_count++ == 0){
				source->additional_branches = malloc(sizeof(neuromorph_node*));
			}
			else{
				source->additional_branches = realloc(source->additional_branches, sizeof(neuromorph_node*)*source->additional_branch_count);
			}
			source->additional_branches[source->additional_branch_count-1] = destination;
			return 1;
		}
		source->next = destination;
		return 1;
	case OUTPUT_NODE:
	default:
		return 0;
	}
	return 0;
}

uint8_t neuromorph_link_destination(neuromorph_node* const source, neuromorph_node* const destination){
	switch(destination->type){
	case OUTPUT_NODE:
	case LAYER_NODE:
		destination->prev = source;
		if (source->neuron_buffer != NULL){
			destination->weight_buffer_size = source->buffer_size*destination->buffer_size;
		}
		else{
			destination->weight_buffer_size = *source->previous_buffer_size*destination->buffer_size;
		}
		destination->weight_buffer = malloc(sizeof(float)*destination->weight_buffer_size);
		return 1;
	case DIVERGENT_NODE:
		neuromorph_information_transfer_destination_link(source, destination);
		return 1;
	case CONVERGENT_NODE:
		if (destination->prev == NULL){
			neuromorph_information_transfer_destination_link(source, destination);
			if (destination->neuron_buffer == NULL){
				destination->buffer_size = *destination->previous_buffer_size;
				destination->neuron_buffer = malloc(sizeof(float)*destination->buffer_size);
			}
			return 1;
		}
		if (source->neuron_buffer != NULL){
			destination->convergent_buffer = source->neuron_buffer;
			destination->convergent_buffer_size = &source->buffer_size;
			return 1;
		}
		destination->convergent_buffer = source->previous_neuron_buffer;
		destination->convergent_buffer_size = source->previous_buffer_size;
		return 1;
	case INPUT_NODE:
	default:
		return 0;
	}
	return 0;
}

void neuromorph_information_transfer_destination_link(neuromorph_node* const source, neuromorph_node* const destination){
	destination->prev = source->prev;
	if (source->neuron_buffer != NULL){
		destination->previous_neuron_buffer = source->neuron_buffer;
		destination->previous_buffer_size = &source->buffer_size;
	}
	else{
		destination->previous_neuron_buffer = source->previous_neuron_buffer;
		destination->previous_buffer_size = source->previous_buffer_size;
	}
}

void neuromorph_input(neuromorph_node* const node){
	//TODO
}

void neuromorph_divergent(neuromorph_node* const node){
	//TODO
}

void neuromorph_convergent(neuromorph_node* const node){
	//TODO
}

void neuromorph_layer(neuromorph_node* const node){
	//TODO
}

neuromorph* neuromorph_init(){
	neuromorph* model = malloc(sizeof(neuromorph));
	model->adjacency = adjacency_map_init();
	model->input = NULL;
	return model;
}

void neuromorph_free(neuromorph* model){
	adjacency_map_iterator it = adjacency_map_iterator_init(&model->adjacency);
	while (adjacency_map_iterator_has_next(&it)){
		vector v = adjacency_map_iterator_next(&it).val;
		for (uint16_t i = 0;i<v.size;++i) neuromorph_node_free((neuromorph_node*)v.data[i]);
		vector_free(&v);
	}
	adjacency_map_free(&model->adjacency);
	free(model);
}

void neuromorph_build(neuromorph* model, const char* const description){
	// TODO
}

static PyObject* helloworld(PyObject* self, PyObject* args){
	return Py_BuildValue("s", "Hello, World!");
}

static PyObject* say_hello(PyObject* self, PyObject* args){
	const char* name;
	if (!PyArg_ParseTuple(args, "s", &name)){
		return NULL;
	}
	char greeting[256];
	snprintf(greeting, sizeof(greeting), "Hello, %s", name);
	return Py_BuildValue("s", greeting);
}

static PyMethodDef NeuroMorph[] = {
	{"helloworld",(PyCFunction)helloworld,METH_NOARGS, "Prints Hello, World!"},
	{"say_hello",(PyCFunction)say_hello,METH_VARARGS, "Greets the given argument"},
	{NULL,NULL,0,NULL}
};

PyMODINIT_FUNC PyInit_neuromorph(void){
	static struct PyModuleDef neuromorphmodule = {
		PyModuleDef_HEAD_INIT,
		"neuromorph",
		NULL,
		-1,
		NeuroMorph
	};
	return PyModule_Create(&neuromorphmodule);
}
