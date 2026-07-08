#ifndef MLXKIT_C_H
#define MLXKIT_C_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*mlxkit_completion_callback)(
    void *context,
    int32_t status,
    const char *error_message
);

typedef void (*mlxkit_token_callback)(void *context, const char *token);

void *mlxkit_runtime_create(void);
void mlxkit_runtime_destroy(void *runtime);

int32_t mlxkit_runtime_load_model(
    void *runtime,
    const char *directory,
    mlxkit_completion_callback completion,
    void *context
);

int32_t mlxkit_runtime_load_model_id(
    void *runtime,
    const char *model_id,
    const char *revision,
    mlxkit_completion_callback completion,
    void *context
);

int32_t mlxkit_runtime_chat(
    void *runtime,
    const char *messages_json,
    float temperature,
    int32_t max_tokens,
    mlxkit_token_callback on_token,
    mlxkit_completion_callback completion,
    void *context
);

#ifdef __cplusplus
}
#endif

#endif
