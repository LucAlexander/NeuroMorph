#ifndef NEUROMORPH_H
#define NEUROMORPH_H

#ifdef nm_fma
#include <immintrin.h>
#endif
#include <smmintrin.h>
#include <pthread.h>
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

#define ACTIVATION_TYPE void (*)(float* const, const size_t, const float)
#define ACTIVATION_DERIVATIVE_TYPE void (*)(float* const, const float* const, const size_t, const float)
#define LOSS_TYPE float (*)(float* const, const float* const, const float* const, const size_t, const float)
#define LOSS_DERIVATIVE_TYPE void (*)(float* const, const float* const, const float* const, const size_t, const float)
#define GENERIC_FUNCTION_TYPE void* (*)(void*)

typedef struct neuromorph_node{
	struct neuromorph_node* next;
	struct neuromorph_node* prev;
	NEUROMORPH_NODE_TYPE type;
	//TODO this should really just be an 8 bit flag, but we'll leave that for the optimization step
	uint8_t ready; // forward parallel node ready
	uint8_t back_ready; // backward parallel node ready
	uint8_t loop; // node is the last step of a memory link converging to main branch
	uint8_t loop_start; // node is the start of a branch which econverges back in time
	uint8_t unrolled; // indicates that its ok to aggregate the gradient from a loop
	uint8_t unrolled_front; // indicates that its already been propogated through 
	/* Normal buffer
	 * input node uses it as a standard buffer for the initial pass
	 * convergent node uses it as a buffer for convergence between two previous branches
	 * output and layer nodes use it for pass data computation
	*/
	float* neuron_buffer;
	float* neuron_buffer_raw;
	size_t buffer_size;
	// Used by layers and output for weights and biases/ activation function for buffer calculation
	float* weight_buffer;
	size_t weight_buffer_size;
	float* bias_buffer;
	size_t bias_buffer_size;
	void (*activation_function)(float* const buffer, const size_t size, const float parameter);
	void (*activation_function_derivative)(float* const gradient, const float* const buffer, const size_t size, const float parameter);
	float activation_parameter;
	// Used by specialized output node for loss function
	float (*loss_function)(float* const buffer, const float* const result, const float* const expected, const size_t size, const float parameter);
	void (*loss_function_derivative)(float* const gradient, const float* const result, const float* const expected, const size_t size, const float paramaeter);
	float loss_parameter;
	float* expected;
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
	void (*convergence_function)(const float* const branch_buffer, const float* const previous, float* const output_buffer, const size_t size);
	void (*convergence_function_derivative)(const float* const prev_gradient, const float* const prev, const float* const path, float* const gradient, float* const path_gradient, const size_t size);
	pthread_mutex_t mutex;
	pthread_cond_t cond;
	// Used for calculating weight_gradients, previous means input previous still
	const size_t* previous_backlog_offset;
	const size_t* previous_backlog_activation;
	// Used for backpropogation of gradients, previous means output previous
	float* gradient_buffer;
	float* path_gradient_buffer;
	float* previous_weight_buffer;
	float** previous_gradient_buffer;
	const size_t* previous_gradient_size;
	// where out preactivated and activated results are stored in the models memory
	size_t backlog_offset;
	size_t backlog_offset_activation;
}neuromorph_node;

neuromorph_node* neuromorph_input_init(size_t input_size);
neuromorph_node* neuromorph_divergent_init();
neuromorph_node* neuromorph_convergent_init(void (*convergence)(const float* const, const float* const, float* const, const size_t), void(*convergence_derivative)(const float* const, const float* const, const float* const, float* const, float* const, const size_t));
neuromorph_node* neuromorph_layer_init(size_t buffer_size, void (*activation)(float* const, const size_t, const float), void (*activation_derivative)(float* const, const float* const, const size_t, const float), float parameter);
neuromorph_node* neuromorph_output_init(size_t buffer_size, void (*activation)(float* const, const size_t, const float), void (*activation_derivative)(float* const, const float* const, const size_t, const float), float activation_parameter, float (*loss)(float* const, const float* const, const float* const, const size_t, const float), void (*loss_derivative)(float* const, const float* const, const float* const, const size_t, const float), float loss_parameter);

void neuromorph_node_free(neuromorph_node*);

void neuromorph_link(adjacency_map* adjacency, neuromorph_node* const source, neuromorph_node* const destination);
uint8_t neuromorph_link_source(neuromorph_node* const source, neuromorph_node* const destination);
uint8_t neuromorph_link_destination(neuromorph_node* const source, neuromorph_node* const destination);
void neuromorph_information_transfer_destination_link(neuromorph_node* const source, neuromorph_node* const destination);

typedef int64_t ast_node_id;
VECTOR(vector_u64, ast_node_id)

typedef struct neuromorph_layer_args{
	size_t layer_size;
	void (*activation_function)(float* const, const size_t, const float);
	void (*activation_function_derivative)(float* const gradient, const float* const buffer, const size_t size, const float parameter);
	float (*loss_function)(float* const, const float* const, const float* const, const size_t, const float);
	void (*loss_function_derivative)(float* const gradient, const float* const result, const float* const expected, const size_t size, const float paramaeter);
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
	void (*convergence_function)(const float* const, const float* const, float* const, const size_t);
	void (*convergence_function_derivative)(const float* const, const float*, const float* const, float* const, float* const, const size_t);
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

#define BIAS_TYPE void (*)(float* const, const size_t, const float, const float)
#define WEIGHT_TYPE void (*)(float* const, const size_t, const size_t, const float, const float)

typedef struct neuromorph_header{
	void (*bias_function)(float* const, const size_t, const float, const float);
	void (*weight_function)(float* const, const size_t, const size_t, const float, const float);
	float bias_parameter_a;
	float bias_parameter_b;
	float weight_parameter_a;
	float weight_parameter_b;
}neuromorph_header;

typedef struct neuromorph{
	ast_node_id ast_root;
	neuromorph_ast ast;
	neuromorph_header header;
	adjacency_map adjacency;
	neuromorph_node* input;
	neuromorph_node* output;
	uint16_t batch_size;
	float* batch_backlog;
	float* batch_expected;
	size_t backlog_size;
	pthread_mutex_t backlog_mutex;
	float learning_rate;
}neuromorph;

neuromorph* neuromorph_init(size_t batch_size, float learning_rate);
void neuromorph_free(neuromorph* model);
void adjacency_map_free_internal(adjacency_map* adjacency);

#define NODE_NAME_TOKEN_MAX 64

typedef enum PARAMETRIC_FUNCTION_TYPE{
	PARAMETRIC_ACTIVATION,
	PARAMETRIC_LOSS,
	PARAMETRIC_WEIGHT,
	PARAMETRIC_BIAS
}PARAMETRIC_FUNCTION_TYPE;

typedef union parametric_function_data{
	void (*activation)(float* const buffer, const size_t size, const float parameter);
	float (*loss)(float* const buffer, const float* const result, const float* const expected, const size_t buffer_size, const float parameter);
}parametric_function_data;

typedef union parametric_derivative_data{
	void (*activation)(float* const gradient, const float* const buffer, const size_t size, const float parameter);
	void (*loss_function_derivative)(float* const gradient, const float* const result, const float* const expected, const size_t size, const float paramaeter);
}parametric_derivative_data;

typedef struct parametric_function{
	PARAMETRIC_FUNCTION_TYPE type;
	parametric_function_data function;
	parametric_derivative_data derivative;
	float parameter;
}parametric_function;

typedef struct function_record{
	const char* name;
	PARAMETRIC_FUNCTION_TYPE type;
	void* (*function)(void*);
	void* (*function_derivative)(void*);
}function_record;

neuromorph* neuromorph_compile(const char* const description, size_t batch_size, float learning_rate);
uint8_t parse_parametric_function(parametric_function* const func, const char** c);
void ast_converge_branches(neuromorph_ast* const ast);
uint8_t neuromorph_compile_check_legal(neuromorph_ast* const ast, const ast_node_id* const root);
uint8_t neuromorph_layer_check_legal(neuromorph_layer_args layer, const ast_node_id* const next, ast_node_id* const output, ast_node_id* const node_id);
uint8_t neuromorph_divergence_check_legal(neuromorph_divergence_args divergence, const ast_node_id* const root);
uint8_t neuromorph_convergence_check_legal(neuromorph_convergence_args convergence, neuromorph_ast* const ast);
ast_node_id name_to_id(const char* name);
neuromorph_ast_node* neuromorph_ast_set_next_id(neuromorph_ast* ast, ast_node_id id, ast_node_id next);

uint8_t neuromorph_layer_arg_parse(neuromorph_ast_node* node, uint16_t arg_i, const char* const arg);
uint8_t neuromorph_pass_parametric_function(neuromorph_ast_node* const node, uint16_t arg_i, const parametric_function* const param);
uint8_t neuromorph_convergence_arg_parse(neuromorph_ast_node* node, uint16_t arg_i, const char* const arg);
uint8_t neuromorph_divergence_arg_parse(neuromorph_ast_node* node, uint16_t arg_t, ast_node_id id);
uint8_t neuromorph_parse_segment(neuromorph_ast* ast, const char** c, char type, ast_node_id* const id, ast_node_id prev, uint8_t first, uint8_t true_root);
void neuromorph_ast_free_internal(neuromorph_ast* const ast);

HASHMAP(graph_domain, ast_node_id, uintptr_t)
void neuromorph_build(neuromorph* model);
neuromorph_node* neuromorph_build_branch(neuromorph_ast* ast, ast_node_id node_id, adjacency_map* adjacency, graph_domain* domain, uint8_t branch, neuromorph_node* node, size_t* const backlog_size);
void build_divergent_branches(vector* stale_links, vector* div_nodes, vector_u64* divs, neuromorph_ast* ast, graph_domain* domain, adjacency_map* adjacency, neuromorph_node* leftover, size_t* const backlog_size);
void register_backlog(neuromorph_node* current_node, size_t* const backlog_size);
uint8_t neuromorph_mark_loops(neuromorph_node* node, vector* marked);
neuromorph_node* neuromorph_pull_output(adjacency_map* map);

#define GELU_C 0.044715

void convergence_multiplicative(const float* const path, const float* const previous, float* const buffer, const size_t buffer_size);
void convergence_additive(const float* const path, const float* const previous, float* const buffer, const size_t buffer_size);
void convergence_average(const float* const path, const float* const previous, float* const buffer, const size_t buffer_size);
//TODO concatenation, attention, weight, billinear matrix?

#define PARAMETRIC_FUNCTION_COUNT 18

float loss_mse(float* const buffer, const float* const result, const float* const expected, const size_t size, const float parameter);
float loss_mae(float* const buffer, const float* const result, const float* const expected, const size_t size, const float parameter);
float loss_mape(float* const buffer, const float* const result, const float* const expected, const size_t size, const float parameter);
float loss_huber(float* const buffer, const float* const result, const float* const expected, const size_t size, const float parameter);
float loss_huber_modified(float* const buffer, const float* const result, const float* const expected, const size_t size, const float parameter);
float loss_cross_entropy(float* const buffer, const float* const result, const float* const expected, const size_t size, const float parameter);
float loss_hinge(float* const buffer, const float* const result, const float* const expected, const size_t size, const float parameter);

#ifdef nm_sse
__m128 exp_neg_ps(__m128 x);
__m128 tanh_ps(__m128 x);
#endif

void activation_sigmoid(float* const buffer, const size_t size, const float parameter);
void activation_relu(float* const buffer, const size_t size, const float parameter);
void activation_tanh(float* const buffer, const size_t size, const float parameter);
void activation_binary_step(float* const buffer, const size_t size, const float parameter);
void activation_linear(float* const buffer, const size_t size, const float parameter);
void activation_relu_leaky(float* const buffer, const size_t size, const float parameter);
void activation_relu_parametric(float* const buffer, const size_t size, const float parameter);
void activation_elu(float* const buffer, const size_t size, const float parameter);
void activation_softmax(float* const buffer, const size_t size, const float parameter);
void activation_swish(float* const buffer, const size_t size, const float parameter);
void activation_gelu(float* const buffer, const size_t size, const float parameter);
void activation_selu(float* const buffer, const size_t size, const float parameter);

typedef struct forward_args{
	neuromorph_node* node;
	float* backlog;
	pthread_mutex_t* mut;
	uint16_t batch;
} forward_args;

float neuromorph_forward(neuromorph* model, uint16_t pass_count);
uint8_t end_of_branch(neuromorph_node* node);
void thread_signal_ready(neuromorph_node* node);
void write_to_backlog(float* const backlog, pthread_mutex_t* mut, const float* const buffer, const size_t size, const size_t offset, uint16_t batch);
void node_pass(neuromorph_node* node);
void* neuromorph_branch_forward(void* args);

void set_seed(time_t seed);
float uniform_distribution(float min, float max);
float normal_distribution(float mean, float std);

#define PARAMETRIC_INITIALIZATION_COUNT 9

void bias_initialization_zero(float* const buffer, const size_t size, const float a, const float b);
void bias_initialization_const_flat(float* const buffer, const size_t size, const float a, const float b);
void bias_initialization_const_uneven(float* const buffer, const size_t size, const float a, const float b);

void weight_initialization_xavier(float* const out, const size_t in_size, const size_t out_size, const float a, const float b);
void weight_initialization_he(float* const out, const size_t in_size, const size_t out_size, const float a, const float b);
void weight_initialization_lecun(float* const out, const size_t in_size, const size_t out_size, const float a, const float b);
void weight_initialization_uniform(float* const out, const size_t in_size, const size_t out_size, const float a, const float b);
void weight_initialization_orthogonal(float* const out, const size_t in_size, const size_t out_size, const float a, const float b);
void weight_initialization_normal(float* const out, const size_t in_size, const size_t out_size, const float a, const float b);

void weight_bias_initialize(neuromorph* model);
neuromorph_header compile_header(const char** c);
uint8_t parse_header_function(neuromorph_header* header, const char* token, uint8_t arg_c);
uint8_t parse_header_parameter(neuromorph_header* header, float* bias_parameter, float* weight_parameter, const char* token);

void convergence_multiplicative_partial(const float* const prev_gradient, const float* const prev, const float* const path, float* const gradient, float* const path_gradient, const size_t size);
void convergence_additive_partial(const float* const prev_gradient, const float* const prev, const float* const path, float* const gradient, float* const path_gradient, const size_t size);
void convergence_average_partial(const float* const prev_gradient, const float* const prev, const float* const path, float* const gradient, float* const path_gradient, const size_t size);

void loss_mse_partial(float* const gradient, const float* const result, const float* const expected, const size_t size, const float parameter);
void loss_mae_partial(float* const gradient, const float* const result, const float* const expected, const size_t size, const float parameter);
void loss_mape_partial(float* const gradient, const float* const result, const float* const expected, const size_t size, const float parameter);
void loss_huber_partial(float* const gradient, const float* const result, const float* const expected, const size_t size, const float parameter);
void loss_huber_modified_partial(float* const gradient, const float* const result, const float* const expected, const size_t size, const float parameter);
void loss_cross_entropy_partial(float* const gradient, const float* const result, const float* const expected, const size_t size, const float parameter);
void loss_hinge_partial(float* const gradient, const float* const result, const float* const expected, const size_t size, const float parameter);

void activation_sigmoid_partial(float* const gradient, const float* const buffer, const size_t size, const float parameter);
void activation_relu_partial(float* const gradient, const float* const buffer, const size_t size, const float parameter);
void activation_tanh_partial(float* const gradient, const float* const buffer, const size_t size, const float parameter);
void activation_binary_step_partial(float* const gradient, const float* const buffer, const size_t size, const float parameter);
void activation_linear_partial(float* const gradient, const float* const buffer, const size_t size, const float parameter);
void activation_relu_leaky_partial(float* const gradient, const float* const buffer, const size_t size, const float parameter);
void activation_relu_parametric_partial(float* const gradient, const float* const buffer, const size_t size, const float parameter);
void activation_elu_partial(float* const gradient, const float* const buffer, const size_t size, const float parameter);
void activation_softmax_partial(float* const gradient, const float* const buffer, const size_t size, const float parameter);
void activation_swish_partial(float* const gradient, const float* const buffer, const size_t size, const float parameter);
void activation_gelu_partial(float* const gradient, const float* const buffer, const size_t size, const float parameter);
void activation_selu_partial(float* const gradient, const float* const buffer, const size_t size, const float parameter);

typedef struct backprop_args{
	neuromorph_node* node;
	size_t batch_size;
	size_t backlog_size;
	float* backlog;
	float* expected_backlog;
	float learning_rate;
} backprop_args;

void neuromorph_back(neuromorph* model);
void* neuromorph_branch_back(void* args);
void update_learnables(neuromorph_node* node, size_t batch_size, float learning_rate, float* weight_gradients);
void gradient_propogate_end(neuromorph_node* node, size_t backlog_size, size_t batch_size, float* backlog, float* expected_backlog, float learning_rate);
void gradient_propogate(neuromorph_node* node, size_t backlog_size, size_t batch_size, float* backlog, float learning_rate);
void neuromorph_train_batch(neuromorph* model, float* input, float* expected, uint8_t verbose);
void back_transfer_logic(neuromorph_node* node, size_t batch_size, size_t backlog_size, float* backlog, float* expected_backlog, float learning_rate);
void aggregate_diverged_gradients(neuromorph_node* node);
void construct_base_gradients_layer(neuromorph_node* node, float* base_gradients);
void construct_base_gradients_divergence(neuromorph_node* node, float* base_gradients);
void update_learnables(neuromorph_node* node, size_t batch_size, float learning_rate, float* weight_gradients);

#endif
