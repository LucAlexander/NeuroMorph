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
	node->activation_parameter = 0;
	node->loss_function = NULL;
	node->loss_parameter = 0;
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

neuromorph_node* neuromorph_convergent_init(void (*convergence)(const float* const, float* const, const size_t)){
	neuromorph_node* node = neuromorph_divergent_init();
	node->convergence_function = convergence;
	node->type = CONVERGENT_NODE;
	return node;
}

neuromorph_node* neuromorph_layer_init(size_t buffer_size, void (*activation)(float* const, const size_t, const float), float parameter){
	neuromorph_node* node = neuromorph_input_init(buffer_size);
	node->bias_buffer = malloc(sizeof(float)*buffer_size);
	node->bias_buffer_size = buffer_size;
	node->activation_function = activation;
	node->activation_parameter = parameter;
	node->type = LAYER_NODE;
	return node;
}

neuromorph_node* neuromorph_output_init(size_t buffer_size, void (*activation)(float* const, const size_t, const float), float activation_parameter, float (*loss)(float* const, const float* const, const float* const, const size_t, const float), float loss_parameter){
	neuromorph_node* node = neuromorph_layer_init(buffer_size, activation, activation_parameter);
	node->loss_function = loss;
	node->loss_parameter = loss_parameter;
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
		if (!neuromorph_parse_segment(&ast, &c, *c, &id, prev, first, first)){
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
	if (arg_i != 0){
		fprintf(stderr, "additional non function argument passed to layer\n");
		return 0;
	}
	node->data.layer.layer_size = atoi(arg);
	if (node->data.layer.layer_size == 0){
		fprintf(stderr, "%s not a valid layer size\n", arg);
		return 0;
	}
	return 1;
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
		else if (!strcmp(arg, "average")){
			node->data.convergence.convergence_function = convergence_average;
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

uint8_t evaluate_parametric_function_name(parametric_function* const func, const char* const name){
	function_record function_list[] = {
		{"sigmoid", PARAMETRIC_ACTIVATION, (GENERIC_FUNCTION_TYPE)activation_sigmoid},
		{"relu", PARAMETRIC_ACTIVATION, (GENERIC_FUNCTION_TYPE)activation_relu},
		{"relu_leaky", PARAMETRIC_ACTIVATION, (GENERIC_FUNCTION_TYPE)activation_relu_leaky},
		{"tanh", PARAMETRIC_ACTIVATION, (GENERIC_FUNCTION_TYPE)activation_tanh},
		{"softmax", PARAMETRIC_ACTIVATION, (GENERIC_FUNCTION_TYPE)activation_softmax},
		{"elu", PARAMETRIC_ACTIVATION, (GENERIC_FUNCTION_TYPE)activation_elu},
		{"gelu", PARAMETRIC_ACTIVATION, (GENERIC_FUNCTION_TYPE)activation_gelu},
		{"swish", PARAMETRIC_ACTIVATION, (GENERIC_FUNCTION_TYPE)activation_swish},
		{"relu_parametric", PARAMETRIC_ACTIVATION, (GENERIC_FUNCTION_TYPE)activation_relu_parametric},
		{"selu", PARAMETRIC_ACTIVATION, (GENERIC_FUNCTION_TYPE)activation_selu},
		{"linear", PARAMETRIC_ACTIVATION, (GENERIC_FUNCTION_TYPE)activation_linear},
		{"binary_step", PARAMETRIC_ACTIVATION, (GENERIC_FUNCTION_TYPE)activation_binary_step},
		{"mse", PARAMETRIC_LOSS, (GENERIC_FUNCTION_TYPE)loss_mse},
		{"mae", PARAMETRIC_LOSS, (GENERIC_FUNCTION_TYPE)loss_mae},
		{"mape", PARAMETRIC_LOSS, (GENERIC_FUNCTION_TYPE)loss_mape},
		{"huber", PARAMETRIC_LOSS, (GENERIC_FUNCTION_TYPE)loss_huber},
		{"hinge", PARAMETRIC_LOSS, (GENERIC_FUNCTION_TYPE)loss_hinge},
		{"cross_entropy", PARAMETRIC_LOSS, (GENERIC_FUNCTION_TYPE)loss_cross_entropy}
	};
	for (size_t i = 0;i<PARAMETRIC_FUNCTION_COUNT;++i){
		if (!strcmp(name, function_list[i].name)){
			func->type = function_list[i].type;
			if (func->type == PARAMETRIC_ACTIVATION){
				func->function.activation = (ACTIVATION_TYPE)function_list[i].function;
				return 1;
			}
			func->function.loss = (LOSS_TYPE)function_list[i].function;
			return 1;
		}
	}
	fprintf(stderr, "not a valid parametric function name\n");
	return 0;
}

uint8_t parse_parametric_function(parametric_function* const func, const char** c){
	char arg[NODE_NAME_TOKEN_MAX];
	uint16_t index = 0;
	uint8_t argument_set = 0;
	for (++(*c);**c!='\0';++(*c)){
		switch(**c){
		case ' ':
		case '\n':
		case '\r':
		case '\t':
		case '\b':
			break;
		case ')':
		case ']':
		case '}':
		case '|':
		case '<':
		case '[':
		case '{':
		case '(':
			fprintf(stderr, "unexpected token in parametric function parse %c\n", **c);
			return 0;
		case '>':
			arg[index] = '\0';
			if (!argument_set){
				if (!evaluate_parametric_function_name(func, arg)){
					fprintf(stderr,"missing parametric function name, recieved only %s\n", arg);
					return 0;
				}
			}
			func->parameter = atof(arg);
			return 1;
		case ',':
			arg[index] = '\0';
			index = 0;
			argument_set = 1;
			if (!evaluate_parametric_function_name(func, arg)){
				return 0;
			}
			break;
		default:
			if (index == NODE_NAME_TOKEN_MAX-1){
				fprintf(stderr, "invalid parametric argument, token length exceeded\n");
				return 0;
			}
			arg[index++] = **c;
			break;
		}
	}
	fprintf(stderr, "unexpected end of description during function parse\n");
	return 0;
}

uint8_t neuromorph_pass_parametric_function(neuromorph_ast_node* const node, uint16_t arg_i, const parametric_function* const param){
	if (param->type == PARAMETRIC_ACTIVATION){
		node->data.layer.activation_function = (ACTIVATION_TYPE)param->function.activation;
		node->data.layer.activation_parameter = param->parameter;
		return 1;
	}
	if (param->type == PARAMETRIC_LOSS){
		node->data.layer.loss_function = (LOSS_TYPE)param->function.loss;
		node->data.layer.loss_parameter = param->parameter;
		return 1;
	}
	fprintf(stderr, "parametric function passed in wrong argument position\n");
	return 0;
}

uint8_t neuromorph_parse_segment(neuromorph_ast* ast, const char** c, char type, ast_node_id* const top_level_id, ast_node_id prev, uint8_t first, uint8_t true_root){
	neuromorph_ast_node node;
	uint8_t found_start = 0;
	for (;!found_start;++(*c)){
		switch(**c){
		case '\0':
			return 1;
		case '(':
			found_start = 1;
			node.type = NEUROMORPH_LAYER_ARGS;
			node.data.layer.layer_size = 0;
			node.data.layer.activation_function = NULL;
			node.data.layer.loss_function = NULL;
			node.data.layer.input = true_root;
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
	parametric_function param = {0, {NULL}, 0};
	uint8_t function_arg = 0;
	for ((*c)++;**c!='\0';++(*c)){
		switch (**c){
		case ' ':
		case '\n':
		case '\t':
		case '\r':
		case '\b':
			break;
		case '>':
			fprintf(stderr, "unexpected token, no function to end\n");
			return 0;
		case '<':
			if (node.type != NEUROMORPH_LAYER_ARGS){
				fprintf(stderr, "parametric function passed in non layer\n");
				return 0;
			}
			if (!parse_parametric_function(&param, c)){
				fprintf(stderr, "invalid parametric activation function\n");
				return 0;
			}
			function_arg = 1;
			break;
		case '(':
		case '{':
		case '[':
			ast_node_id temp_id;
			if (!neuromorph_parse_segment(ast, c, **c, &temp_id, sub_prev, branch_start, 0)){
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
			if (function_arg){
				if (!neuromorph_pass_parametric_function(&node, arg_i++, &param)){
					fprintf(stderr, "parametric activation passed in incorrect argument sloot\n");
					return 0;
				}
				function_arg = 0;
			}
			else if (index != 0){
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
				if (function_arg){
					if (!neuromorph_pass_parametric_function(&node, arg_i++, &param)){
						fprintf(stderr, "parametric function passed in incorrect argument slot\n");
						return 0;
					}
					function_arg = 0;
					break;
				}
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
	if (layer.activation_function == NULL && !layer.input){
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
	if (convergence.path == -1){
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
					current_node = neuromorph_layer_init(ast_node->data.layer.layer_size, ast_node->data.layer.activation_function, ast_node->data.layer.activation_parameter);
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
				current_node = neuromorph_output_init(ast_node->data.layer.layer_size, ast_node->data.layer.activation_function, ast_node->data.layer.activation_parameter, ast_node->data.layer.loss_function, ast_node->data.layer.loss_parameter);
				break;
			}
			current_node = neuromorph_layer_init(ast_node->data.layer.layer_size, ast_node->data.layer.activation_function, ast_node->data.layer.activation_parameter);
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

/*
 * 1. x86_64 Architecture:

    SSE: While older, SSE is still relevant and provides a baseline of SIMD support for nearly all x86_64 CPUs.
    AVX: Covers many modern but not the latest CPUs. AVX and AVX2 are commonly supported and offer significant performance improvements for many tasks.
    AVX-512: Found in some high-end Intel CPUs and Xeon processors. It's especially relevant for HPC and data center environments.

2. ARM Architecture:

    NEON: If you later decide to support ARM, NEON is the SIMD technology for ARM processors. It’s widely supported in ARMv7 and ARMv8 architectures.

3. GPU Acceleration:

    CUDA: For NVIDIA GPUs. A significant portion of machine learning, especially deep learning, is accelerated using NVIDIA GPUs.
    OpenCL/ROCm: For AMD GPUs and other accelerators. OpenCL is a framework for writing programs that execute across heterogeneous platforms.

Feasible Steps for Implementation:

    Baseline Implementation:
        Begin with a scalar (non-vectorized) baseline implementation that works on any architecture.

    x86_64 Optimizations:
        Implement optimizations for SSE, AVX, and AVX-512. This can be done incrementally, starting with the most commonly supported instruction sets (SSE and AVX).

    GPU Acceleration (Optional):
        Depending on the nature of your machine learning tasks, consider implementing GPU-accelerated versions using CUDA or OpenCL. GPUs are prevalent in the machine learning field.

    ARM Support (Future Expansion):
        When you’re ready to expand, add support for ARM with NEON optimizations. This will be especially relevant for mobile devices, some servers, and Apple Silicon Macs.

    Library Utilization:
        Consider using libraries like Intel MKL, OpenBLAS, or cuBLAS for matrix operations, which are already optimized for various architectures.

    Testing and Validation:
        Ensure comprehensive testing on different architectures to validate performance and correctness.

Summarized Strategy:

    Short Term: Focus on x86_64, covering SSE through AVX-512. Consider GPU acceleration if relevant for your tasks.
    Medium Term: Evaluate the need for ARM support based on your user base and application requirements. Implement ARM NEON optimizations if needed.
    Long Term: Keep an eye on emerging trends. For example, RISC-V is gaining traction, and the landscape of accelerators (GPUs, TPUs, FPGAs, etc.) is evolving.
 */
//TODO vectorize everything!!!!!
//look into data prefetching when simulation is done
//alignment regardless of version
//more efficient approximations of complex functions
#if defined(sse)
void convergence_multiplicative(const float* const path, float* const buffer, const size_t buffer_size){
	size_t i;
	for (i = 0;i<=buffer_size-4;i+=4){
		__m128 p = _mm_loadu_ps(path+i);
		__m128 b = _mm_loadu_ps(buffer+i);
		__m128 result = _mm_mul_ps(p, b);
		_mm_storeu_ps(buffer+i, result);
	}
	for (;i<buffer_size;++i){
		buffer[i] *= path[i];
	}
}

void convergence_additive(const float* const path, float* const buffer, const size_t buffer_size){
	size_t i;
	for(i = 0;i<=buffer_size-4;i+=4){
		__m128 p = _mm_loadu_ps(path+i);
		__m128 b = _mm_loadu_ps(buffer+i);
		__m128 result = _mm_add_ps(p, b);
		_mm_storeu_ps(buffer + i, result);
	}
	for (;i<buffer_size;++i){
		buffer[i] += path[i];
	}
}

void convergence_average(const float* const path, float* const buffer, const size_t buffer_size){
	size_t i;
	__m128 two = _mm_set1_ps(2.0f);
	for (i = 0;i<=buffer_size-4;i+=4){
		__m128 p = _mm_loadu_ps(path+i);
		__m128 b = _mm_loadu_ps(buffer+i);
		__m128 s = _mm_add_ps(p, b);
		__m128 avg = _mm_div_ps(s, two);
		_mm_storeu_ps(buffer+i, avg);
	}
	for (;i<buffer_size;++i){
		buffer[i] = (buffer[i]+path[i])/2;
	}
}

float loss_mse(float* const buffer, const float* const result, const float* const expected, const size_t size, const float parameter){
	__m128 s = _mm_setzero_ps();
	size_t i;
	for (i= 0;i<=size-4;i+=4){
		__m128 e = _mm_loadu_ps(expected+i);
		__m128 r = _mm_loadu_ps(result+i);
		__m128 loss = _mm_sub_ps(e, r);
		_mm_storeu_ps(buffer+i,loss);
#ifdef fma
		s = __fmadd_ps(loss, loss, s);
#else
		__m128 loss_squared = _mm_mul_ps(loss, loss);
		s = _mm_add_ps(s,loss_squared);
#endif
	}
	float sum_array[4];
	_mm_storeu_ps(sum_array, s);
	float sum = sum_array[0]+sum_array[1]+sum_array[2]+sum_array[3];
	for (;i<size;++i){
		float loss = expected[i]-result[i];
		buffer[i] = loss;
		sum += loss*loss;
	}
	return (sum)/(size);
}

float loss_mae(float* const buffer, const float* const result, const float* const expected, const size_t size, const float parameter){
	__m128 s = _mm_setzero_ps();
	size_t i;
	for (i= 0;i<=size-4;i+=4){
		__m128 e = _mm_loadu_ps(expected+i);
		__m128 r = _mm_loadu_ps(result+i);
		__m128 loss = _mm_sub_ps(e, r);
		_mm_storeu_ps(buffer+i,loss);
		__m128 abs_loss = _mm_andnot_ps(_mm_set1_ps(-0.f), loss);
		s = _mm_add_ps(s,abs_loss);
	}
	float sum_array[4];
	_mm_storeu_ps(sum_array, s);
	float sum = sum_array[0]+sum_array[1]+sum_array[2]+sum_array[3];
	for (;i<size;++i){
		float loss = expected[i]-result[i];
		buffer[i] = loss;
		sum += fabsf(loss);
	}
	return sum/(size);
}

float loss_mape(float* const buffer, const float* const result, const float* const expected, const size_t size, const float parameter){
	__m128 s = _mm_setzero_ps();
	size_t i;
	for (i= 0;i<=size-4;i+=4){
		__m128 e = _mm_loadu_ps(expected+i);
		__m128 r = _mm_loadu_ps(result+i);
		__m128 loss = _mm_sub_ps(e, r);
		_mm_storeu_ps(buffer+i,loss);
		__m128 abs_loss = _mm_andnot_ps(_mm_set1_ps(-0.f), loss);
		__m128 loss_div_e = _mm_div_ps(abs_loss, e);
		s = _mm_add_ps(s,loss_div_e);
	}
	float sum_array[4];
	_mm_storeu_ps(sum_array, s);
	float sum = sum_array[0]+sum_array[1]+sum_array[2]+sum_array[3];
	for (;i<size;++i){
		float expect = expected[i];
		float loss = expect-result[i];
		buffer[i] = loss;
		sum += fabsf(loss/expect);
	}
	return sum/(size);
}

float loss_huber(float* const buffer, const float* const result, const float* const expected, const size_t size, const float parameter){
	__m128 s = _mm_setzero_ps();
	__m128 param_sq_half = _mm_set1_ps(parameter*parameter*0.5f);
	__m128 param = _mm_set1_ps(parameter);
	__m128 half = _mm_set1_ps(0.5f);
	size_t i;
	for (i= 0;i<=size-4;i+=4){
		__m128 e = _mm_loadu_ps(expected+i);
		__m128 r = _mm_loadu_ps(result+i);
		__m128 loss = _mm_sub_ps(e, r);
		_mm_storeu_ps(buffer+i,loss);
		__m128 abs_loss = _mm_andnot_ps(_mm_set1_ps(-0.f), loss);
		__m128 mask = _mm_cmple_ps(abs_loss, param);
		__m128 case1 = _mm_mul_ps(_mm_mul_ps(loss, loss), half);
#ifdef fma
		__m128 case2 = _mm_fmsub_ps(param, abs_loss, param_sq_half);
#else
		__m128 case2 = _mm_sub_ps(_mm_mul_ps(param, abs_loss), param_sq_half);
#endif
#ifdef sse4_1
		__m128 combined = _mm_blendv_ps(case2, case1, mask);
#else
		__m128 combined = _mm_or_ps(_mm_and_ps(mask, case1), _mm_andnot_ps(mask, case2));
#endif
		s = _mm_add_ps(s,combined);
	}
	float sum_array[4];
	_mm_storeu_ps(sum_array, s);
	float sum = sum_array[0]+sum_array[1]+sum_array[2]+sum_array[3];
	float hpsq = parameter*parameter*0.5;
	for (;i<size;++i){
		float expect = expected[i];
		float res = result[i];
		float x = expect-res;
		buffer[i] = x;
		if (abs(x) <= parameter){
			sum += x*x*0.5;
			continue;
		}
		sum += (parameter*fabsf(x))-hpsq;
	}
	return sum;
}

float loss_huber_modified(float* const buffer, const float* const result, const float* const expected, const size_t size, const float parameter){
	__m128 s = _mm_setzero_ps();
	size_t i;
	for (i= 0;i<=size-4;i+=4){
		__m128 e = _mm_loadu_ps(expected+i);
		__m128 r = _mm_loadu_ps(result+i);
		__m128 loss = _mm_sub_ps(e, r);
		_mm_storeu_ps(buffer+i,loss);
		__m128 prod = _mm_mul_ps(e, r);
		__m128 mask = _mm_cmpgt_ps(prod, _mm_set1_ps(-1.f));
		__m128 ones = _mm_set1_ps(1.f);
		__m128 sub = _mm_sub_ps(ones, prod);
		__m128 zeros = _mm_setzero_ps();
		__m128 case1 = _mm_max_ps(zeros, sub);
		case1 = _mm_mul_ps(case1, case1);
		__m128 case2 = _mm_mul_ps(_mm_set1_ps(-4.f), prod);
#ifdef sse4_1
		__m128 combined = _mm_blendv_ps(case2, case1, mask);
#else
		__m128 combined = _mm_or_ps(_mm_and_ps(mask, case1), _mm_andnot_ps(mask, case2));
#endif
		s = _mm_add_ps(s,combined);
	}
	float sum_array[4];
	_mm_storeu_ps(sum_array, s);
	float sum = sum_array[0]+sum_array[1]+sum_array[2]+sum_array[3];
	for (;i<size;++i){
		float expect = expected[i];
		float res = result[i];
		float x = expect-res;
		buffer[i] = x;
		x = expect*res;
		if (x > -1){
			sum += powf(fmaxf(0, 1-x),2);
			continue;
		}
		sum -= 4*x;
	}
	return sum;
}

float loss_cross_entropy(float* const buffer, const float* const result, const float* const expected, const size_t size, const float parameter){
	__m128 s = _mm_setzero_ps();
	size_t i;
	for (i= 0;i<=size-4;i+=4){
		__m128 e = _mm_loadu_ps(expected+i);
		__m128 r = _mm_loadu_ps(result+i);
		__m128 loss = _mm_sub_ps(e, r);
		_mm_storeu_ps(buffer+i,loss);
		__m128 log_r = _mm_set_ps(logf(r[3]), logf(r[2]), logf(r[1]), logf(r[0]));
		__m128 term = _mm_mul_ps(e, log_r);
		s = _mm_add_ps(s,term);
	}
	float sum_array[4];
	_mm_storeu_ps(sum_array, s);
	float sum = sum_array[0]+sum_array[1]+sum_array[2]+sum_array[3];
	for (;i<size;++i){
		float expect = expected[i];
		float res = result[i];
		buffer[i] = expect-res;
		sum += expect*logf(res);
	}
	return -sum;
}

float loss_hinge(float* const buffer, const float* const result, const float* const expected, const size_t size, const float parameter){
	__m128 s = _mm_setzero_ps();
	size_t i;
	for (i= 0;i<=size-4;i+=4){
		__m128 e = _mm_loadu_ps(expected+i);
		__m128 r = _mm_loadu_ps(result+i);
		__m128 loss = _mm_sub_ps(e, r);
		_mm_storeu_ps(buffer+i,loss);
		__m128 prod = _mm_mul_ps(e, r);
		__m128 ones = _mm_set1_ps(1.0f);
		__m128 term = _mm_sub_ps(ones, prod);
		__m128 zeros = _mm_setzero_ps();
		__m128 hinge = _mm_max_ps(zeros, term);
		s = _mm_add_ps(s,hinge);
	}
	float sum_array[4];
	_mm_storeu_ps(sum_array, s);
	float sum = sum_array[0]+sum_array[1]+sum_array[2]+sum_array[3];
	for (;i<size;++i){
		float expect = expected[i];
		float res = result[i];
		buffer[i] = expect-res;
		sum += fmaxf(0,1-(expect*res));
	}
	return sum;
}

__m128 exp_neg_ps(__m128 x) {
	const __m128 one = _mm_set1_ps(1.0f);
	const __m128 exp_c1 = _mm_set1_ps(0.04166669f);
	const __m128 exp_c2 = _mm_set1_ps(0.5000004f);
	__m128 x2 = _mm_mul_ps(x, x);  // x^2
	__m128 x3 = _mm_mul_ps(x2, x); // x^3
	// Compute the polynomial approximation: e^-x ≈ 1 - x - x^2/2 - x^3/6
	__m128 poly = _mm_sub_ps(one, x);        // 1 - x
	poly = _mm_sub_ps(poly, _mm_mul_ps(exp_c2, x2)); // 1 - x - x^2/2
	poly = _mm_sub_ps(poly, _mm_mul_ps(exp_c1, x3)); // 1 - x - x^2/2 - x^3/6
	return poly;
}

void activation_sigmoid(float* const buffer, const size_t size, const float parameter){
	size_t i;
	const __m128 one = _mm_set1_ps(1.0f);
	for (i = 0;i<=size-4;i+=4){
		__m128 x = _mm_loadu_ps(buffer+1);
		x = _mm_xor_ps(x, _mm_set1_ps(-0.f));
		__m128 exp_neg_x = exp_neg_ps(x);
		__m128 sigmoid = _mm_div_ps(one, _mm_add_ps(one, exp_neg_x));
		_mm_storeu_ps(buffer+i, sigmoid);
	}
	for (;i<size;++i){
		buffer[i] = 1/(1+expf(-buffer[i]));
	}
}

void activation_relu(float* const buffer, const size_t size, const float parameter){
	size_t i;
	for (i = 0;i<=size-4;i+=4){
		__m128 zeros = _mm_setzero_ps();
		__m128 term = _mm_loadu_ps(buffer+i);
		__m128 relu = _mm_max_ps(zeros, term);
		_mm_storeu_ps(buffer+i, relu);
	}
	for (;i<size;++i){
		buffer[i] = fmaxf(0,buffer[i]);
	}
}

__m128 tanh_ps(__m128 x) {
    const __m128 one = _mm_set1_ps(1.0f);
    // Calculate e^(2x)
    __m128 exp2x = exp_neg_ps(_mm_mul_ps(x, _mm_set1_ps(-2.0f)));
    exp2x = _mm_rcp_ps(exp2x);
    // Calculate (e^(2x) - 1) / (e^(2x) + 1)
    __m128 num = _mm_sub_ps(exp2x, one);
    __m128 den = _mm_add_ps(exp2x, one);
    __m128 tanh_x = _mm_div_ps(num, den);
    return tanh_x;
}

void activation_tanh(float* const buffer, const size_t size, const float parameter){
	size_t i;
	for (i = 0;i<=size-4;i+=4){
		__m128 x = _mm_loadu_ps(buffer+i);
		__m128 tanh_approx = tanh_ps(x);
		_mm_storeu_ps(buffer+i, tanh_approx);
	}
	for (;i<size;++i){
		buffer[i] = tanh(buffer[i]);
	}
}

void activation_binary_step(float* const buffer, const size_t size, const float parameter){
	const __m128 zero = _mm_setzero_ps();
	size_t i;
	for (i = 0;i<=size-4;i+=4){
		__m128 x = _mm_loadu_ps(buffer+i);
		__m128 step = _mm_cmpge_ps(x, zero);
		_mm_storeu_ps(buffer+i, step);
	}
	for (;i<size;++i){
		buffer[i] = buffer[i] >= 0;
	}
}

void activation_linear(float* const buffer, const size_t size, const float parameter){
	return;
}

void activation_relu_leaky(float* const buffer, const size_t size, const float parameter){
	size_t i;
	const __m128 tenth = _mm_set1_ps(0.1f);
	for (i=0;i<=size-4;i+=4){
		__m128 x = _mm_loadu_ps(buffer+i);
		__m128 term = _mm_max_ps(_mm_mul_ps(tenth, x), x);
		_mm_storeu_ps(buffer+i, term);
	}
	for (;i<size;++i){
		float x = buffer[i];
		buffer[i] = fmaxf(0.1*x,x);
	}
}

void activation_relu_parametric(float* const buffer, const size_t size, const float parameter){
	size_t i;
	const __m128 tenth = _mm_set1_ps(parameter);
	for (i=0;i<=size-4;i+=4){
		__m128 x = _mm_loadu_ps(buffer+i);
		__m128 term = _mm_max_ps(_mm_mul_ps(tenth, x), x);
		_mm_storeu_ps(buffer+i, term);
	}
	for (;i<size;++i){
		float x = buffer[i];
		buffer[i] = fmaxf(parameter*x,x);
	}
}

void activation_elu(float* const buffer, const size_t size, const float parameter){
	const __m128 zero = _mm_setzero_ps();
	const __m128 one = _mm_set1_ps(1.0f);
	const __m128 alpha = _mm_set1_ps(parameter);
	size_t i;
	for (i=0;i<=size-4;i+=4){
		__m128 x = _mm_loadu_ps(buffer+i);
		__m128 mask = _mm_cmplt_ps(x, zero);
#ifdef sse4_1
#ifdef fma
		__m128 negs = _mm_fmadd_ps(alpha, exp_neg_ps(x), _mm_set1_ps(-alpha));
#else
		__m128 negs = _mm_mul_ps(alpha, _mm_sub_ps(exp_neg_ps(x), one));
#endif
		__m128 term = _mm_blendv_ps(x, negs, mask);
#else
#ifdef fma
		__m128 negs = _mm_and_ps(mask, _mm_fmadd_ps(alpha, exp_neg_ps(x), _mm_set1_ps(-alpha)));
#else
		__m128 negs = _mm_and_ps(mask, _mm_mul_ps(alpha, _mm_sub_ps(exp_neg_ps(x), one)));
#endif
		__m128 term = _mm_add_ps(_mm_andnot_ps(mask, x), negs);
#endif
		_mm_storeu_ps(buffer+i, term);
	}
	for (;i<size;++i){
		float x = buffer[i];
		if (x < 0){
			buffer[i] = parameter*(expf(x)-1);
		}
	}
}

void activation_softmax(float* const buffer, const size_t size, const float parameter){
	__m128 s = _mm_setzero_ps();
	const __m128 n1 = _mm_set1_ps(-1.0f);
	size_t i;
	for (i = 0;i<=size-4;i+=4){
		__m128 x = _mm_loadu_ps(buffer+i);
		__m128 exp_x = exp_neg_ps(_mm_mul_ps(x, n1));
		s = _mm_add_ps(s, exp_x);
	}
	float simd_s[4];
	_mm_storeu_ps(simd_s, s);
	float denom = simd_s[0]+simd_s[1]+simd_s[2]+simd_s[3];
	for (;i<size;++i){
		denom += expf(buffer[i]);
	}
	const __m128 d = _mm_set1_ps(denom);
	for (i = 0;i<=size-4;i+=4){
		__m128 x = _mm_loadu_ps(buffer+i);
		__m128 exp_x = exp_neg_ps(_mm_mul_ps(x, n1));
		__m128 term = _mm_div_ps(exp_x, d);
		_mm_storeu_ps(buffer+i, term);
	}
	for (;i<size;++i){
		buffer[i] = expf(buffer[i])/denom;
	}
}

void activation_swish(float* const buffer, const size_t size, const float parameter){
	size_t i;
	const __m128 one = _mm_set1_ps(1.0f);
	for (i = 0;i<=size-4;i+=4){
		__m128 x = _mm_loadu_ps(buffer+i);
		__m128 denom = _mm_add_ps(one, exp_neg_ps(x));
		__m128 term = _mm_div_ps(x, denom);
		_mm_storeu_ps(buffer+i, term);
	}
	for (;i<size;++i){
		float x = buffer[i];
		buffer[i] = x/(1+expf(-x));
	}
}

void activation_gelu(float* const buffer, const size_t size, const float parameter){
	size_t i;
	const float s2p = sqrtf(2.0f)/PI;
	const __m128 half = _mm_set1_ps(0.5f);
	const __m128 one = _mm_set1_ps(1.0f);
	const __m128 sqrt2vpi = _mm_set1_ps(s2p);
	const __m128 gelu_c = _mm_set1_ps(GELU_C);
	for (i = 0;i<=size-4;i+=4){
		__m128 x = _mm_loadu_ps(buffer+i);
		__m128 a = _mm_mul_ps(sqrt2vpi, _mm_add_ps(x, _mm_mul_ps(gelu_c, _mm_mul_ps(x, _mm_mul_ps(x, x)))));
		__m128 tanh_a = tanh_ps(a);
		__m128 gelu = _mm_mul_ps(half, _mm_mul_ps(x, _mm_add_ps(one, tanh_a)));
		_mm_storeu_ps(buffer+i, gelu);
	}
	for (;i<size;++i){
		float x = buffer[i];
		buffer[i] = 0.5*x*(1+tanh(s2p*(x+(GELU_C*powf(x,3)))));
	}
}

void activation_selu(float* const buffer, const size_t size, const float parameter){
	//TODO unfortunately not possible with current compiler, will need more compelx syntax logic for describing activation functions, allowing multiple parameters
}

#else
void convergence_multiplicative(const float* const path, float* const buffer, const size_t buffer_size){
	for (size_t i = 0;i<buffer_size;++i){
		buffer[i] *= path[i];
	}
}

void convergence_additive(const float* const path, float* const buffer, const size_t buffer_size){
	for (size_t i = 0;i<buffer_size;++i){
		buffer[i] += path[i];
	}
}

void convergence_average(const float* const path, float* const buffer, const size_t buffer_size){
	for (size_t i = 0;i<buffer_size;++i){
		buffer[i] = (buffer[i]+path[i])/2;
	}
}

float loss_mse(float* const buffer, const float* const result, const float* const expected, const size_t size, const float parameter){
	float sum = 0;
	for (size_t i = 0;i<size;++i){
		float loss = expected[i]-result[i];
		buffer[i] = loss;
		sum += loss*loss;
	}
	return (sum)/(size);
}

float loss_mae(float* const buffer, const float* const result, const float* const expected, const size_t size, const float parameter){
	float sum = 0;
	for (size_t i = 0;i<size;++i){
		float loss = expected[i]-result[i];
		buffer[i] = loss;
		sum += abs(loss);
	}
	return sum/(size);
}

float loss_mape(float* const buffer, const float* const result, const float* const expected, const size_t size, const float parameter){
	float sum = 0;
	for (size_t i = 0;i<size;++i){
		float expect = expected[i];
		float loss = expect-result[i];
		buffer[i] = loss;
		sum += abs(loss/expect);
	}
	return sum/(size);
}

float loss_huber(float* const buffer, const float* const result, const float* const expected, const size_t size, const float parameter){
	float sum = 0;
	float hpsq = parameter*parameter*0.5;
	for (size_t i = 0;i<size;++i){
		float expect = expected[i];
		float res = result[i];
		float x = expect-res;
		buffer[i] = x;
		if (abs(x) <= parameter){
			sum += x*x*0.5;
			continue;
		}
		sum += (parameter*abs(x))-hpsq;
	}
	return sum;
}

float loss_huber_modified(float* const buffer, const float* const result, const float* const expected, const size_t size, const float parameter){
	float sum = 0;
	for (size_t i = 0;i<size;++i){
		float expect = expected[i];
		float res = result[i];
		float x = expect-res;
		buffer[i] = x;
		x = expect*res;
		if (x > -1){
			sum += pow(fmax(0, 1-x),2);
			continue;
		}
		sum -= 4*x;
	}
	return sum;
}

float loss_cross_entropy(float* const buffer, const float* const result, const float* const expected, const size_t size, const float parameter){
	float sum = 0;
	for (size_t i = 0;i<size;++i){
		float expect = expected[i];
		float res = result[i];
		buffer[i] = expect-res;
		sum += expect*log(res);
	}
	return -sum;
}

float loss_hinge(float* const buffer, const float* const result, const float* const expected, const size_t size, const float parameter){
	float sum = 0;
	for (size_t i = 0;i<size;++i){
		float expect = expected[i];
		float res = result[i];
		buffer[i] = expect-res;
		sum += fmax(0,1-(expect*res));
	}
	return sum;
}

void activation_sigmoid(float* const buffer, const size_t size, const float parameter){
	for (size_t i = 0;i<size;++i){
		buffer[i] = 1/(1+expf(-buffer[i]));
	}
}

void activation_relu(float* const buffer, const size_t size, const float parameter){
	for (size_t i = 0;i<size;++i){
		buffer[i] = fmax(0,buffer[i]);
	}
}

void activation_tanh(float* const buffer, const size_t size, const float parameter){
	for (size_t i = 0;i<size;++i){
		buffer[i] = tanh(buffer[i]);
	}
}

void activation_binary_step(float* const buffer, const size_t size, const float parameter){
	for (size_t i = 0;i<size;++i){
		buffer[i] = buffer[i] >= 0;
	}
}

void activation_linear(float* const buffer, const size_t size, const float parameter){
	return;
}

void activation_relu_leaky(float* const buffer, const size_t size, const float parameter){
	float x;
	for (size_t i = 0;i<size;++i){
		x = buffer[i];
		buffer[i] = fmax(0.1*x,x);
	}
}

void activation_relu_parametric(float* const buffer, const size_t size, const float parameter){
	float x;
	for (size_t i = 0;i<size;++i){
		x = buffer[i];
		buffer[i] = fmax(parameter*x, x);
	}
}

void activation_elu(float* const buffer, const size_t size, const float parameter){
	float x;
	for (size_t i = 0;i<size;++i){
		x = buffer[i];
		if (x < 0){
			buffer[i] = parameter*(expf(x)-1);
		}
	}
}

void activation_softmax(float* const buffer, const size_t size, const float parameter){
	float denom = 0;
	for (size_t i = 0;i<size;++i){
		denom += expf(buffer[i]);
	}
	for (size_t i = 0;i<size;++i){
		buffer[i] = expf(buffer[i])/denom;
	}
}

void activation_swish(float* const buffer, const size_t size, const float parameter){
	float x;
	for (size_t i = 0;i<size;++i){
		x = buffer[i];
		buffer[i] = x/(1+expf(-x));
	}
}

void activation_gelu(float* const buffer, const size_t size, const float parameter){
	float x;
	const float s2p  = sqrtf(2/PI);
	for (size_t i = 0;i<size;++i){
		x = buffer[i];
		buffer[i] = 0.5*x*(1+tanh(s2p*(x+(GELU_C*pow(x,3)))));
	}
}

void activation_selu(float* const buffer, const size_t size, const float parameter){
	//TODO unfortunately not possible with current compiler, will need more compelx syntax logic for describing activation functions, allowing multiple parameters
}
#endif

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
	const char* description;
	if (!PyArg_ParseTuple(args, "s", &description)){
		return NULL;
	}
	neuromorph* model = neuromorph_compile(description);
	neuromorph_build(model);
	neuromorph_free(model);
	char greeting[256];
	snprintf(greeting, sizeof(greeting), "compiled and built description:\n\n%s", description);
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
