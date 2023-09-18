#ifndef NEUROMORPH_H
#define NEUROMORPH_H

#include <inttypes.h>
#include "hashmap.h"
#include "vector.h"

VECTOR(vector, uintptr_t)
HASHMAP(adjacency_map, uintptr_t, vector)

typedef enum NEUROMORPH_NODE_TYPE{
	INPUT_NODE,
	DIVERGENT_NODE,
	CONVERGENT_NODE,
	LAYER_NODE,
	OUTPUT_NODE
}NEUROMORPH_NODE_TYPE;

typedef struct neuromorph_node{
	struct neuromorph_node* next;
	struct neuromorph_node* prev;
	NEUROMORPH_NODE_TYPE type;
	uint8_t ready;
	/* Normal buffer
	 * input node uses it as a standard buffer for the initial pass
	 * convergent node uses it as a buffer for convergence between two previous branches
	 * output and layer nodes use it for pass data computation
	*/
	float* neuron_buffer;
	size_t buffer_size;
	// Used by layers and output for weights and biases/ activation function for buffer calculation
	float* weight_buffer;
	size_t weight_buffer_size;
	float* bias_buffer;
	size_t bias_buffer_size;
	void (*activation_function)(float* const buffer, const size_t* const size);
	// Used by specialized output node for loss function
	void (*loss_function)(float* const buffer, const size_t* const size);
	// Used by information flow nodes to keep track of the previuos layer, convergence,  or input buffer
	const float* previous_neuron_buffer;
	const size_t* previous_buffer_size;
	// Used by Divergent nodes to keep track of additional next pointers for branches
	struct neuromorph_node** additional_branches;
	size_t additional_branch_count;
	// Used by Convergent nodes to keep track of converging branches and their neuron buffer data
	struct neuromorph_node* convergent_node;
	const float* convergent_buffer;
	const size_t* convergent_buffer_size;
	void (*convergence_function)(const float* const branch_buffer, float* const output_buffer, const size_t* const size);
}neuromorph_node;

neuromorph_node* neuromorph_input_init(size_t input_size);
neuromorph_node* neuromorph_divergent_init();
neuromorph_node* neuromorph_convergent_init(void (*convergence)(const float* const, float* const, const size_t* const));
neuromorph_node* neuromorph_layer_init(size_t buffer_size, void (*activation)(float* const, const size_t* const));
neuromorph_node* neuromorph_output_init(size_t buffer_size, void (*activation)(float* const, const size_t* const), void (*loss)(float* const, const size_t* const));

void neuromorph_node_free(neuromorph_node*);

void neuromorph_link(adjacency_map* adjacency, neuromorph_node* const source, neuromorph_node* const destination);
uint8_t neuromorph_link_source(neuromorph_node* const source, neuromorph_node* const destination);
uint8_t neuromorph_link_destination(neuromorph_node* const source, neuromorph_node* const destination);
void neuromorph_information_transfer_destination_link(neuromorph_node* const source, neuromorph_node* const destination);

void neuromorph_input(neuromorph_node* const node);
void neuromorph_divergent(neuromorph_node* const node);
void neuromorph_convergent(neuromorph_node* const node);
void neuromorph_layer(neuromorph_node* const node);

typedef struct neuromorph{
	adjacency_map adjacency;
	neuromorph_node* input;
}neuromorph;

neuromorph* neuromorph_init();
void neuromorph_free(neuromorph* model);

void neuromorph_build(neuromorph* model, const char* const description);

#endif
