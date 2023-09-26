#include <Python.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

#include "NeuroMorph.h"

VECTOR_SOURCE(vector, uintptr_t)
HASHMAP_SOURCE(adjacency_map, uintptr_t, vector, hash_i)
VECTOR_SOURCE(vector_u64, ast_node_id)
HASHMAP_SOURCE(neuromorph_ast, ast_node_id, neuromorph_ast_node, hash_i)
HASHMAP_SOURCE(graph_domain, ast_node_id, uintptr_t, hash_i)

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
	node->type = CONVERGENT_NODE;
	return node;
}

neuromorph_node* neuromorph_layer_init(size_t buffer_size, void (*activation)(float* const, const size_t* const)){
	neuromorph_node* node = neuromorph_input_init(buffer_size);
	node->bias_buffer = malloc(sizeof(float)*buffer_size);
	node->bias_buffer_size = buffer_size;
	node->activation_function = activation;
	node->type = LAYER_NODE;
	return node;
}

neuromorph_node* neuromorph_output_init(size_t buffer_size, void (*activation)(float* const, const size_t* const), float (*loss)(float* const, const float* const, const float* const, const size_t* const)){
	neuromorph_node* node = neuromorph_layer_init(buffer_size, activation);
	node->loss_function = loss;
	node->type = OUTPUT_NODE;
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
	if (!v){
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

void adjacency_map_free_internal(adjacency_map* adjacency){
	adjacency_map_iterator it = adjacency_map_iterator_init(adjacency);
	while (adjacency_map_iterator_has_next(&it)){
		adjacency_map_result r = adjacency_map_iterator_next(&it);
		neuromorph_node_free((neuromorph_node*)r.key);
		vector_free(&r.val);
	}
	adjacency_map_free(adjacency);
}

void neuromorph_free(neuromorph* model){
	adjacency_map_free_internal(&model->adjacency);
	neuromorph_ast_free_internal(&model->ast);
	free(model);
}

neuromorph* neuromorph_compile(const char* const description){
	neuromorph_ast ast = neuromorph_ast_init();
	if (*description != '('){
		fprintf(stderr, "Model must start with input node as layer denored by (args)\n, found unexpected token: %c", *description);
		return NULL;
	}
	const char* c = description;
	ast_node_id id;
	ast_node_id prev = -1;
	ast_node_id root = -1;
	uint8_t first = 1;
	while (*c != '\0'){
		if (!neuromorph_parse_segment(&ast, &c, *c, &id, prev, first)){
			fprintf(stderr, "compile ended early\n");
			neuromorph_ast_free_internal(&ast);
			return NULL;
		}
		if (first){
			root = id;
		}
		prev = id;
		first = 0;
	}
	ast_converge_branches(&ast);
	if (!neuromorph_compile_check_legal(&ast, &root)){
		fprintf(stderr, "not a legal graph\n");
		neuromorph_ast_free_internal(&ast);
		return NULL;
	}
	neuromorph* model = neuromorph_init();
	model->ast = ast;
	model->ast_root = root;
	return model;
}

void ast_converge_branches(neuromorph_ast* const ast){
	neuromorph_ast_iterator it = neuromorph_ast_iterator_init(ast);
	while (neuromorph_ast_iterator_has_next(&it)){
		neuromorph_ast_result r = neuromorph_ast_iterator_next(&it);
		ast_node_id me = r.key;
		neuromorph_ast_node node = r.val;
		if (node.type == NEUROMORPH_CONVERGENCE_ARGS){
			ast_node_id path = node.data.convergence.path;
			neuromorph_ast_set_next_id(ast, path, me);
		}
	}
}

ast_node_id name_to_id(const char* arg){
	ast_node_id hash = 5381;
	int16_t c;
	while((c=*arg++)) hash = ((hash << 5) + hash) + c;
	return hash;
}

uint8_t neuromorph_layer_arg_parse(neuromorph_ast_node* node, uint16_t arg_i, const char* const arg){
	switch(arg_i){
	case 0:
		node->data.layer.layer_size = atoi(arg);
		return 1;
	case 1:
		if (!strcmp(arg, "sigmoid")){
			node->data.layer.activation_function = activation_sigmoid;
		}
		else if (!strcmp(arg, "relu")){
			node->data.layer.activation_function = activation_relu;
		}
		else if (!strcmp(arg, "tanh")){
			node->data.layer.activation_function = activation_tanh;
		}
		else{
			fprintf(stderr, "unknown activation function %s\n", arg);
			return 0;
		}
		return 1;
	case 2:
		if (!strcmp(arg, "mse")){
			node->data.layer.loss_function = loss_mse;
		}
		else{
			fprintf(stderr, "unknnown loss function %s\n", arg);
			return 0;
		}
		return 1;
	}
	return 0;
}

uint8_t neuromorph_convergence_arg_parse(neuromorph_ast_node* node, uint16_t arg_i, const char* const arg){
	if (arg_i == 0){
		node->data.convergence.path = name_to_id(arg);
		return 1;
	}
	if (arg_i == 1){
		if (!strcmp(arg, "multiplicative")){
			node->data.convergence.convergence_function = convergence_multiplicative;
		}
		else if (!strcmp(arg, "additive")){
			node->data.convergence.convergence_function = convergence_additive;
		}
		else{
			fprintf(stderr, "unknown convergence function %s\n", arg);
			return 0;
		}
		return 1;
	}
	fprintf(stderr, "too many arguments for convergence node\n");
	return 0;
}

uint8_t neuromorph_divergence_arg_parse(neuromorph_ast_node* node, uint16_t arg_i, ast_node_id id){
	if (arg_i == 0){
		node->data.divergence.paths = vector_u64_init();
		node->data.divergence.initialized = 1;
	}
	vector_u64_push(&node->data.divergence.paths, id);
	return 1;
}

uint8_t neuromorph_ast_set_next_id(neuromorph_ast* ast, ast_node_id id, ast_node_id next){
	neuromorph_ast_node* node = neuromorph_ast_ref(ast, id);
	if (!node){
		fprintf(stderr, "previous node with passed id does not exist\n");
		return 0;
	}
	node->next = next;
	return 1;
}

uint8_t neuromorph_parse_segment(neuromorph_ast* ast, const char** c, char type, ast_node_id* const top_level_id, ast_node_id prev, uint8_t first){
	neuromorph_ast_node node;
	uint8_t found_start = 0;
	for (;!found_start;++(*c)){
		if (**c=='\0'){
			fprintf(stderr, "expression terminated early\n");
			return 0;
		}
		switch(type){
		case '(':
			found_start = 1;
			node.type = NEUROMORPH_LAYER_ARGS;
			node.data.layer.layer_size = 0;
			node.data.layer.activation_function = NULL;
			node.data.layer.loss_function = NULL;
			break;
		case '{':
			found_start = 1;
			node.type = NEUROMORPH_CONVERGENCE_ARGS;
			node.data.convergence.path = -1;
			node.data.convergence.convergence_function = NULL;
			break;
		case '[':
			found_start = 1;
			node.type = NEUROMORPH_DIVERGENCE_ARGS;
			node.data.divergence.initialized = 0;
			break;
		default:
			break;
		}
	}
	node.next = -1;
	char arg[NODE_NAME_TOKEN_MAX];
	uint16_t index = 0;
	for (;**c!=',';++(*c)){
		if (index == NODE_NAME_TOKEN_MAX-1){
			fprintf(stderr, "node name exceeds token length, truncated\n");
			break;
		}
		arg[index++] = **c;
	}
	arg[index] = '\0';
	ast_node_id id = name_to_id(arg);
	*top_level_id = id;
	if ((!first) && prev != -1){
		if (!neuromorph_ast_set_next_id(ast, prev, id)){
			fprintf(stderr, "next reference unable to be set");
			return 0;
		}
	}
	index = 0;
	uint16_t arg_i = 0;
	uint8_t branch_start = 1;
	ast_node_id branch_start_id = -1;
	ast_node_id sub_prev = id;
	for ((*c)++;**c!='\0';++(*c)){
		switch (**c){
		case ' ':
		case '\n':
		case '\t':
		case '\r':
		case '\b':
			break;
		case '(':
		case '{':
		case '[':
			ast_node_id temp_id;
			if (!neuromorph_parse_segment(ast, c, **c, &temp_id, sub_prev, branch_start)){
				return 0;
			}
			--(*c);
			sub_prev = temp_id;
			if (branch_start){
				branch_start_id = temp_id;
				branch_start = 0;
			}
			break;
		case ')':
			if (node.type != NEUROMORPH_LAYER_ARGS){
				fprintf(stderr, "node terminated with unexpected token %c\n", **c);
				return 0;
			}
			if (index != 0){
				arg[index] = '\0';
				if (!neuromorph_layer_arg_parse(&node, arg_i++, arg)){
					fprintf(stderr, "argument error\n");
					return 0;
				}
			}
			++(*c);
			neuromorph_ast_push(ast, id, node);
			return 1;
		case '}':
			if (node.type != NEUROMORPH_CONVERGENCE_ARGS){
				fprintf(stderr, "node terminated with unexpected token %c\n", **c);
				return 0;
			}
			if (index != 0){
				arg[index] = '\0';
				if (!neuromorph_convergence_arg_parse(&node, arg_i++, arg)){
					fprintf(stderr, "argument error\n");
					return 0;
				}
			}
			++(*c);
			neuromorph_ast_push(ast, id, node);
			return 1;
		case ']':
			if (node.type != NEUROMORPH_DIVERGENCE_ARGS){
				fprintf(stderr, "node terminated with unexpected token %c\n", **c);
				return 0;
			}
			if (!branch_start){
				if (!neuromorph_divergence_arg_parse(&node, arg_i++, branch_start_id)){
					fprintf(stderr, "argument error\n");
					return 0;
				}
			}
			++(*c);
			neuromorph_ast_push(ast, id, node);
			return 1;
		case ',':
			arg[index] = '\0';
			index = 0;
			if (node.type == NEUROMORPH_LAYER_ARGS){
				if (!neuromorph_layer_arg_parse(&node, arg_i++, arg)){
					fprintf(stderr, "argument error\n");
					return 0;
				}
				break;
			}
			if (node.type == NEUROMORPH_CONVERGENCE_ARGS){
				if (!neuromorph_convergence_arg_parse(&node, arg_i++, arg)){
					fprintf(stderr, "argument error\n");
					return 0;
				}
				break;
			}
			fprintf(stderr, "argument passed in divergence node\n");
			return 0;
		case '|':
			if (node.type != NEUROMORPH_DIVERGENCE_ARGS){
				fprintf(stderr, "split provided in non divergence\n");
				return 0;
			}
			if (!neuromorph_divergence_arg_parse(&node, arg_i++, branch_start_id)){
				fprintf(stderr, "branch argument error\n");
				return 0;
			}
			branch_start = 1;
			break;
		default:
			if (index == NODE_NAME_TOKEN_MAX-1){
				fprintf(stderr, "argument exceeded length limit\n");
				break;
			}
			arg[index++] = **c;
			break;
		}
	}
	fprintf(stderr, "unexpected end of model\n");
	return 0;
}

void neuromorph_ast_free_internal(neuromorph_ast* const ast){
	neuromorph_ast_iterator it = neuromorph_ast_iterator_init(ast);
	while (neuromorph_ast_iterator_has_next(&it)){
		neuromorph_ast_node node = neuromorph_ast_iterator_next(&it).val;
		if (node.type==NEUROMORPH_DIVERGENCE_ARGS){
			if (node.data.divergence.initialized){
				vector_u64_free(&node.data.divergence.paths);
			}
		}
	}
	neuromorph_ast_free(ast);
}

uint8_t neuromorph_layer_check_legal(neuromorph_layer_args layer, const ast_node_id* const next, ast_node_id* const output, ast_node_id* const node_id){
	if (layer.layer_size == 0){
		fprintf(stderr, "layer size not given\n");
		return 0;
	}
	if (layer.activation_function == NULL){
		fprintf(stderr, "activation function not provided\n");
		return 0;
	}
	if (layer.loss_function != NULL){
		if (*output != -1){
			fprintf(stderr, "multiple output nodes defined\n");
			return 0;
		}
		if (*next != -1){
			fprintf(stderr, "loss function provided to non terminal node\n");
			return 0;
		}
		*output = *node_id;
	}
	else if (*next == -1){
		fprintf(stderr, "non terminal node has no next\n");
		return 0;
	}
	return 1;
}

uint8_t neuromorph_divergence_check_legal(neuromorph_divergence_args divergence, const ast_node_id* const root){
	for (size_t i = 0;i<divergence.paths.size;++i){
		ast_node_id next = divergence.paths.data[i];
		if (next == -1 || next == *root){
			fprintf(stderr, "invalid split path to root or output\n");
			return 0;
		}
	}
	return 1;
}

uint8_t neuromorph_convergence_check_legal(neuromorph_convergence_args convergence, neuromorph_ast* const ast){
	if (convergence.path <= 0){
		fprintf(stderr, "invalid convergence path\n");
		return 0;
	}
	if (convergence.convergence_function == NULL){
		fprintf(stderr, "no convergence function given\n");
		return 0;
	}
	return 1;
}

uint8_t neuromorph_compile_check_legal(neuromorph_ast* const ast, const ast_node_id* const root){
	if (*root == -1){
		fprintf(stderr, "no root\n");
		return 0;
	}
	ast_node_id output = -1;
	neuromorph_ast_iterator it = neuromorph_ast_iterator_init(ast);
	while (neuromorph_ast_iterator_has_next(&it)){
		neuromorph_ast_result r = neuromorph_ast_iterator_next(&it);
		ast_node_id node_id = r.key;
		neuromorph_ast_node ast_node = r.val;
		if (ast_node.next == *root){
			fprintf(stderr, "node points directly to input node\n");
			return 0;
		}
		switch(ast_node.type){
		case NEUROMORPH_LAYER_ARGS:
			if (!neuromorph_layer_check_legal(ast_node.data.layer, &ast_node.next, &output, &node_id)){
				return 0;
			}
			break;
		case NEUROMORPH_CONVERGENCE_ARGS:
			if (!neuromorph_convergence_check_legal(ast_node.data.convergence, ast)){
				return 0;
			}
			break;
		case NEUROMORPH_DIVERGENCE_ARGS:
			if (!neuromorph_divergence_check_legal(ast_node.data.divergence, root)){
				return 0;
			}
			break;
		}
	}
	return 1;
}

void neuromorph_build(neuromorph* model){
	ast_node_id node_id = model->ast_root;
	graph_domain domain = graph_domain_init();
	model->input = neuromorph_build_branch(&model->ast, node_id, &model->adjacency, &domain, 0, NULL);
	graph_domain_free(&domain);
}

neuromorph_node* neuromorph_build_branch(neuromorph_ast* ast, ast_node_id node_id, adjacency_map* adjacency, graph_domain* domain, uint8_t branch, neuromorph_node* node){
	neuromorph_node* initial = NULL;
	uint8_t first = 1;
	while (node_id != -1){
		neuromorph_ast_node* ast_node = neuromorph_ast_ref(ast, node_id);
		neuromorph_node* current_node = NULL;
		switch(ast_node->type){
		case NEUROMORPH_LAYER_ARGS:
			if (first){
				if (branch){
					current_node = neuromorph_layer_init(ast_node->data.layer.layer_size, ast_node->data.layer.activation_function);
				}
				else{
					current_node = neuromorph_input_init(ast_node->data.layer.layer_size);
				}
				initial = current_node;
				first = 0;
				break;
			}
			if (node->type == LAYER_NODE || node->type == INPUT_NODE){
				neuromorph_node* link = neuromorph_divergent_init();
				neuromorph_link(adjacency, node, link);
				node = link;
			}
			if (ast_node->next == -1){
				current_node = neuromorph_output_init(ast_node->data.layer.layer_size, ast_node->data.layer.activation_function, ast_node->data.layer.loss_function);
				break;
			}
			current_node = neuromorph_layer_init(ast_node->data.layer.layer_size, ast_node->data.layer.activation_function);
			break;
		case NEUROMORPH_CONVERGENCE_ARGS:
			current_node = neuromorph_convergent_init(ast_node->data.convergence.convergence_function);
			if (first){
				first = 0;
				initial = current_node;
			}
			break;
		case NEUROMORPH_DIVERGENCE_ARGS:
			current_node = neuromorph_divergent_init();
			if (first){
				first = 0;
				initial = current_node;
			}
			if (node != NULL){
				neuromorph_link(adjacency, node, current_node);
				node = NULL;
			}
			if (!ast_node->data.divergence.initialized){
				break;
			}
			for (size_t i = 0;i<ast_node->data.divergence.paths.size;++i){
				ast_node_id candidate = ast_node->data.divergence.paths.data[i];
				uintptr_t* branch_ptr = graph_domain_ref(domain, candidate);
				if (branch_ptr == NULL){
					neuromorph_build_branch(ast, candidate, adjacency, domain, 1, current_node);
				}
			}
			break;
		}
		if (node != NULL){
			neuromorph_link(adjacency, node, current_node);
		}
		node = current_node;
		graph_domain_push(domain, node_id, (uintptr_t)current_node);
		node_id = ast_node->next;
		if (node_id != -1){
			uintptr_t* next_node_ptr = graph_domain_ref(domain, node_id);
			if (next_node_ptr != NULL){
				neuromorph_node* next_node = (neuromorph_node*)(*next_node_ptr);
				neuromorph_link(adjacency, current_node, next_node);
				return initial;
			}
		}	
	}
	return initial;
}

//TODO vectorize everything!!!!!
//TODO apparently the overhead from dereferencing is larger than I realized, might be best to keep scalar arguments as copies so that they can be stuffed in a register
void convergence_multiplicative(const float* const path, float* const buffer, const size_t* const buffer_size){}
void convergence_additive(const float* const path, float* const buffer, const size_t* const buffer_size){}

float loss_mse(float* const buffer, const float* const result, const float* const expected, const size_t* const size){
	float sum = 0;
	for (size_t i = 0;i<*size;++i){
		float loss = expected[i]-result[i];
		buffer[i] = loss;
		sum += loss*loss;
	}
	return (sum)/(*size);
}

float loss_mae(float* const buffer, const float* const result, const float* const expected, const size_t* const size){
	float sum = 0;
	for (size_t i = 0;i<*size;++i){
		float loss = expected[i]-result[i];
		buffer[i] = loss;
		sum += abs(loss);
	}
	return sum/(*size);
}

float loss_mape(float* const buffer, const float* const result, const float* const expected, const size_t* const size){
	float sum = 0;
	for (size_t i = 0;i<*size;++i){
		float expect = expected[i];
		float loss = expect-result[i];
		buffer[i] = loss;
		sum += abs(loss/expect);
	}
	return sum/(*size);
}

float loss_huber(float* const buffer, const float* const result, const float* const expected, const size_t* const size){
	//TODO unfortunately not possible with current compiler, will need more compelx syntax logic for describing activation functions, allowing additional parameters
	return 1;
}

float loss_cross_entropy(float* const buffer, const float* const result, const float* const expected, const size_t* const size){
	float sum = 0;
	for (size_t i = 0;i<*size;++i){
		float expect = expected[i];
		float res = result[i];
		buffer[i] = expect-res;
		sum += expect*log(res);
	}
	return -sum;
}

float loss_hinge(float* const buffer, const float* const result, const float* const expected, const size_t* const size){
	float sum = 0;
	for (size_t i = 0;i<*size;++i){
		float expect = expected[i];
		float res = result[i];
		buffer[i] = expect-res;
		sum += fmax(0,1-(expect*res));
	}
	return sum;
}

void activation_sigmoid(float* const buffer, const size_t* const size){
	for (size_t i = 0;i<*size;++i){
		buffer[i] = 1/(1+pow(EULER, -buffer[i]));
	}
}

void activation_relu(float* const buffer, const size_t* const size){
	for (size_t i = 0;i<*size;++i){
		buffer[i] = fmax(0,buffer[i]);
	}
}

void activation_tanh(float* const buffer, const size_t* const size){
	for (size_t i = 0;i<*size;++i){
		buffer[i] = tanh(buffer[i]);
	}
}

void activation_binary_step(float* const buffer, const size_t* const size){
	for (size_t i = 0;i<*size;++i){
		buffer[i] = buffer[i] >= 0;
	}
}

void activation_linear(float* const buffer, const size_t* const size){
	return;
}

void activation_relu_leaky(float* const buffer, const size_t* const size){
	float x;
	for (size_t i = 0;i<*size;++i){
		x = buffer[i];
		buffer[i] = fmax(0.1*x,x);
	}
}

void activation_relu_parametric(float* const buffer, const size_t* size){
	//TODO unfortunately not possible with current compiler, will need more compelx syntax logic for describing activation functions, allowing additional parameters
}

void activation_elu(float* const buffer, const size_t* size){
	//TODO unfortunately not possible with current compiler, will need more compelx syntax logic for describing activation functions, allowing additional parameters
}

void activation_softmax(float* const buffer, const size_t* size){
	float denom = 0;
	for (size_t i = 0;i<*size;++i){
		denom += pow(EULER, buffer[i]);
	}
	for (size_t i = 0;i<*size;++i){
		buffer[i] = pow(EULER, buffer[i])/denom;
	}
}

void activation_swish(float* const buffer, const size_t* size){
	float x;
	for (size_t i = 0;i<*size;++i){
		x = buffer[i];
		buffer[i] = x/(1+pow(EULER,-x));
	}
}

void activation_gelu(float* const buffer, const size_t* size){
	float x;
	for (size_t i = 0;i<*size;++i){
		x = buffer[i];
		buffer[i] = 0.5*x*(1+tanh(sqrt(2/PI)*(x+(GELU_C*pow(x,3)))));
	}
}

void activation_selu(float* const buffer, const size_t* size){
	//TODO unfortunately not possible with current compiler, will need more compelx syntax logic for describing activation functions, allowing additional parameters
}

static PyObject* helloworld(PyObject* self, PyObject* args){
	neuromorph* model = neuromorph_compile("(input, 64, sigmoid)[divman,(branchyboi, 128, relu)(bb,256,relu){converge2, otherman, additive}(alex, 768, tanh)|(otherman,48,sigmoid)]{convergeman, alex, multiplicative}(output, 12, sigmoid, mse)");
	printf("compiled\n");
	neuromorph_build(model);
	printf("built\n");
	neuromorph_free(model);
	printf("test passed\n");
	model = neuromorph_compile("(i,5,sigmoid){merge, man, multiplicative}(k,6,sigmoid)[split,[man,]](o,2,relu,mse)");
	printf("compiled\n");
	neuromorph_build(model);
	printf("built\n");
	neuromorph_free(model);
	printf("test passed\n");
	model = neuromorph_compile("(i,5,sigmoid){merge, split, multiplicative}(k,6,sigmoid)[split,](o,2,relu,mse)");
	printf("compiled\n");
	neuromorph_build(model);
	printf("built\n");
	neuromorph_free(model);
	printf("test passed\n");
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
