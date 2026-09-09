/**
 * @copyright (c) 2023-2025     Saint Louis University (SLU)
 * @copyright (c) 2023-2025     Massachusetts Institute of Technology (MIT)
 * @copyright (c) 2023-2025     Nvidia Corporation
 * @copyright (c) 2018-2025     King Abdullah University of Science and Technology (KAUST)
 * @copyright (c) 2018-2023     The University of Tennessee and The University of Tennessee Research Foundation
 *                              All rights reserved.
 **/

#ifndef HICMA_KERNEL_TIME_H
#define HICMA_KERNEL_TIME_H

#ifndef PRINT_KERNEL_TIME
#define PRINT_KERNEL_TIME 0
#endif

#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>

#define HICMA_KERNEL_TIME_BUCKETS 16384

typedef struct hicma_kernel_time_entry_s {
    const void *task;
    double start_time;
    struct hicma_kernel_time_entry_s *next;
} hicma_kernel_time_entry_t;

static pthread_mutex_t hicma_kernel_time_lock = PTHREAD_MUTEX_INITIALIZER;
static hicma_kernel_time_entry_t *hicma_kernel_time_table[HICMA_KERNEL_TIME_BUCKETS];

static inline size_t hicma_kernel_time_hash(const void *task)
{
    uintptr_t key = (uintptr_t)task;
    key >>= 4;
    return (size_t)(key % HICMA_KERNEL_TIME_BUCKETS);
}

static inline void hicma_kernel_time_record(const void *task, double start_time)
{
    size_t bucket = hicma_kernel_time_hash(task);
    hicma_kernel_time_entry_t *entry;

    pthread_mutex_lock(&hicma_kernel_time_lock);
    for(entry = hicma_kernel_time_table[bucket]; entry != NULL; entry = entry->next) {
        if(entry->task == task) {
            entry->start_time = start_time;
            pthread_mutex_unlock(&hicma_kernel_time_lock);
            return;
        }
    }

    entry = (hicma_kernel_time_entry_t *)malloc(sizeof(*entry));
    if(entry != NULL) {
        entry->task = task;
        entry->start_time = start_time;
        entry->next = hicma_kernel_time_table[bucket];
        hicma_kernel_time_table[bucket] = entry;
    }
    pthread_mutex_unlock(&hicma_kernel_time_lock);
}

static inline double hicma_kernel_time_take(const void *task, double fallback_start_time)
{
    size_t bucket = hicma_kernel_time_hash(task);
    hicma_kernel_time_entry_t **entryp;
    hicma_kernel_time_entry_t *entry;
    double start_time = fallback_start_time;

    pthread_mutex_lock(&hicma_kernel_time_lock);
    for(entryp = &hicma_kernel_time_table[bucket]; *entryp != NULL; entryp = &(*entryp)->next) {
        entry = *entryp;
        if(entry->task == task) {
            start_time = entry->start_time;
            *entryp = entry->next;
            free(entry);
            break;
        }
    }
    pthread_mutex_unlock(&hicma_kernel_time_lock);

    return start_time;
}

#endif /* HICMA_KERNEL_TIME_H */
