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

#define ACTIVATION_TYPE void (*)(float* const, const size_t* const)
#define LOSS_TYPE float (*)(float* const, const float* const, const float* const, const size_t* const)
#define GENERIC_FUNCTION_TYPE void* (*)(void*)

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
	float activation_parameter;
	// Used by specialized output node for loss function
	float (*loss_function)(float* const buffer, const float* const result, const float* const expected, const size_t* const size);
	float loss_parameter;
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
neuromorph_node* neuromorph_layer_init(size_t buffer_size, void (*activation)(float* const, const size_t* const), float parameter);
neuromorph_node* neuromorph_output_init(size_t buffer_size, void (*activation)(float* const, const size_t* const), float activation_parameter, float (*loss)(float* const, const float* const, const float* const, const size_t* const), float loss_parameter);

void neuromorph_node_free(neuromorph_node*);

void neuromorph_link(adjacency_map* adjacency, neuromorph_node* const source, neuromorph_node* const destination);
uint8_t neuromorph_link_source(neuromorph_node* const source, neuromorph_node* const destination);
uint8_t neuromorph_link_destination(neuromorph_node* const source, neuromorph_node* const destination);
void neuromorph_information_transfer_destination_link(neuromorph_node* const source, neuromorph_node* const destination);

void neuromorph_input(neuromorph_node* const node);
void neuromorph_divergent(neuromorph_node* const node);
void neuromorph_convergent(neuromorph_node* const node);
void neuromorph_layer(neuromorph_node* const node);

typedef int64_t ast_node_id;
VECTOR(vector_u64, ast_node_id)

typedef struct neuromorph_layer_args{
	size_t layer_size;
	void (*activation_function)(float* const, const size_t* const);
	float (*loss_function)(float* const, const float* const, const float* const, const size_t* const);
	float activation_parameter;
	float loss_parameter;
	uint8_t input;
}neuromorph_layer_args;

typedef struct neuromorph_divergence_args{
	uint8_t initialized;
	vector_u64 paths;
}neuromorph_divergence_args;

typedef struct neuromorph_convergence_args{
	ast_node_id path;
	void (*convergence_function)(const float* const, float* const, const size_t* const);
}neuromorph_convergence_args;

typedef enum NEUROMORPH_AST_NODE_TYPE{
	NEUROMORPH_LAYER_ARGS,
	NEUROMORPH_DIVERGENCE_ARGS,
	NEUROMORPH_CONVERGENCE_ARGS
}NEUROMORPH_AST_NODE_TYPE;

typedef union neuromorph_ast_node_data{
	neuromorph_layer_args layer;
	neuromorph_divergence_args divergence;
	neuromorph_convergence_args convergence;
}neuromorph_ast_node_data;

typedef struct neuromorph_ast_node{
	NEUROMORPH_AST_NODE_TYPE type;
	neuromorph_ast_node_data data;
	ast_node_id next;
}neuromorph_ast_node;

HASHMAP(neuromorph_ast, ast_node_id, neuromorph_ast_node)

typedef struct neuromorph{
	ast_node_id ast_root;
	neuromorph_ast ast;
	adjacency_map adjacency;
	neuromorph_node* input;
}neuromorph;

neuromorph* neuromorph_init();
void neuromorph_free(neuromorph* model);
void adjacency_map_free_internal(adjacency_map* adjacency);

#define NODE_NAME_TOKEN_MAX 64

typedef enum PARAMETRIC_FUNCTION_TYPE{
	PARAMETRIC_ACTIVATION,
	PARAMETRIC_LOSS
}PARAMETRIC_FUNCTION_TYPE;

typedef union parametric_function_data{
	void (*activation)(float* const buffer, const size_t* const size);
	float (*loss)(float* const buffer, const float* const result, const float* const expected, const size_t* const buffer_size);
}parametric_function_data;

typedef struct parametric_function{
	PARAMETRIC_FUNCTION_TYPE type;
	parametric_function_data function;
	float parameter;
}parametric_function;

typedef struct function_record{
	const char* name;
	PARAMETRIC_FUNCTION_TYPE type;
	void* (*function)(void*);
}function_record;

neuromorph* neuromorph_compile(const char* const description);
uint8_t parse_parametric_function(parametric_function* const func, const char** c);
void ast_converge_branches(neuromorph_ast* const ast);
uint8_t neuromorph_compile_check_legal(neuromorph_ast* const ast, const ast_node_id* const root);
uint8_t neuromorph_layer_check_legal(neuromorph_layer_args layer, const ast_node_id* const next, ast_node_id* const output, ast_node_id* const node_id);
uint8_t neuromorph_divergence_check_legal(neuromorph_divergence_args divergence, const ast_node_id* const root);
uint8_t neuromorph_convergence_check_legal(neuromorph_convergence_args convergence, neuromorph_ast* const ast);
ast_node_id name_to_id(const char* name);
uint8_t neuromorph_ast_set_next_id(neuromorph_ast* ast, ast_node_id id, ast_node_id next);

uint8_t neuromorph_layer_arg_parse(neuromorph_ast_node* node, uint16_t arg_i, const char* const arg);
uint8_t neuromorph_pass_parametric_function(neuromorph_ast_node* const node, uint16_t arg_i, const parametric_function* const param);
uint8_t neuromorph_convergence_arg_parse(neuromorph_ast_node* node, uint16_t arg_i, const char* const arg);
uint8_t neuromorph_divergence_arg_parse(neuromorph_ast_node* node, uint16_t arg_t, ast_node_id id);
uint8_t neuromorph_parse_segment(neuromorph_ast* ast, const char** c, char type, ast_node_id* const id, ast_node_id prev, uint8_t first, uint8_t true_root);
void neuromorph_ast_free_internal(neuromorph_ast* const ast);

HASHMAP(graph_domain, ast_node_id, uintptr_t)
void neuromorph_build(neuromorph* model);
neuromorph_node* neuromorph_build_branch(neuromorph_ast* ast, ast_node_id node_id, adjacency_map* adjacency, graph_domain* domain, uint8_t branch, neuromorph_node* node);

#define EULER 2.71828
#define PI 3.14159
#define GELU_C 0.044715

void convergence_multiplicative(const float* const path, float* const buffer, const size_t* const buffer_size);
void convergence_additive(const float* const path, float* const buffer, const size_t* const buffer_size);
void convergence_average(const float* const path, float* const buffer, const size_t* const buffer_size);
//TODO concatenation, attention, weight, billinear matrix?

#define PARAMETRIC_FUNCTION_COUNT 18

float loss_mse(float* const buffer, const float* const result, const float* const expected, const size_t* const size);
float loss_mae(float* const buffer, const float* const result, const float* const expected, const size_t* const size);
float loss_mape(float* const buffer, const float* const result, const float* const expected, const size_t* const size);
float loss_huber(float* const buffer, const float* const result, const float* const expected, const size_t* const size);
float loss_cross_entropy(float* const buffer, const float* const result, const float* const expected, const size_t* const size);
float loss_hinge(float* const buffer, const float* const result, const float* const expected, const size_t* const size);

void activation_sigmoid(float* const buffer, const size_t* const size);
void activation_relu(float* const buffer, const size_t* const size);
void activation_tanh(float* const buffer, const size_t* const size);
void activation_binary_step(float* const buffer, const size_t* const size);
void activation_linear(float* const buffer, const size_t* const size);
void activation_relu_leaky(float* const buffer, const size_t* const size);
void activation_relu_parametric(float* const buffer, const size_t* size);
void activation_elu(float* const buffer, const size_t* size);
void activation_softmax(float* const buffer, const size_t* size);
void activation_swish(float* const buffer, const size_t* size);
void activation_gelu(float* const buffer, const size_t* size);
void activation_selu(float* const buffer, const size_t* size);

#endif
