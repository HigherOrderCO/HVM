// IO platform supporting character-based console IO on UNIX-y systems

// This file is designed to be concatenated to the generated runtime

// Protocol
// ---------------------------------------------------------------------
// Program halts with term         Platform will
// -----------------------------   ----------------------------------------
//
//  IO.do_output_chr c rest         1. Print HVM-int c to stdout (as a character)
//                                  2. Replace term with rest
//  (c a fully-evalauted int)       3. Tell runtime to continue
//
//  IO.do_input_chr cont            1. Read a character c from stdin
//                                  2. Replace term with (cont c)
//                                  3. Tell runtime to continue
//
//  IO.done e                       1. Read-back HVM-expression e to C-string
//                                  2. Print to stdout
//                                  3. Tell runtime to halt
//
//  <anything else>             Undefined behaviour.

// Implementation
// --------------

// Programs may not use all the special constructors
// In that case, these #define s make the platfrom still compile

#ifndef _IO_DO__OUTPUT__CHR_
#define _IO_DO__OUTPUT__CHR_ -1
#endif
#ifndef _IO_DO__INPUT__CHR_
#define _IO_DO__INPUT__CHR_ -2
#endif
#ifndef _IO_DONE_
#define _IO_DONE_ -3
#endif


#include <stdio.h>
#include <sys/time.h>

struct PlatState {
    Worker* mem;
};

PlatState* io_setup(Worker* mem) {
    PlatState * state = malloc (sizeof (PlatState));
    state -> mem = mem;
    return state;
}

static char* str_unknown = "???";

char * decode_cid(u64 cid){
    if (cid < id_to_name_size) {
        return id_to_name_data[cid];
    } else {
        return str_unknown;
    }
}

void print_term (FILE * stream, Worker* mem, Ptr ptr) {
    const u64 code_mcap = 256 * 256 * 256; // max code size = 16 MB
    char* code_data = (char*)malloc(code_mcap * sizeof(char));
    assert(code_data);
    readback(
        code_data,
        code_mcap,
        mem,
        ptr,
        id_to_name_data,
        id_to_name_size);
    fprintf(stream, "%s\n", code_data);
    fflush(stream);
    free(code_data);
}

bool fail(PlatState* state, char* msg, Ptr term){
    fprintf(stderr,"[Console IO platform] %s\n",msg);
    print_term(stderr,state->mem,term);
    free (state);
    return false;
}

char* BAD_TOP_MSG = "Illegal IO request!\nExpected one of the constructors {IO.do_output_chr/2, IO.do_input_chr/1, IO.done/0}, instead got:";

bool io_step(PlatState* state) {
    Worker* mem = state->mem;
    Ptr top = mem->node[0];

    if(get_tag(top) != CTR) return fail(state,BAD_TOP_MSG,top);

    switch (get_ext(top)) {

        case _IO_DONE_:
            print_term(stdout,mem, ask_arg(mem,top,1));
            // TODO: decrement IO.done node?
        return false;

        case _IO_DO__OUTPUT__CHR_:
            Ptr num_cell  = ask_arg(mem,top,0);
            Ptr rest_cell = ask_arg(mem,top,1);

            // Step 1: extract character & print
            if(!(get_tag(num_cell) == NUM && get_num(num_cell) <= 256))
                return fail(state,"Bad number cell.\nExpected an int <= 256, instead got:",num_cell);
            printf("%c", (char)get_num(num_cell));

            // Step 2: replace top term with rest
            link(mem, 0, rest_cell);        // Overwrite [CTR|IO.do_output|...] (the root) with [rest_cell], and update backpointers
            // TODO: Free the constructor-data node (two consecutive cells: [num_cell][rest_cell])

        return true;

        case _IO_DO__INPUT__CHR_:
            return fail(state,"Not implemented yet: IO.do_input_chr\nYour term:",top);
        return true;

        default:
        return fail(state,BAD_TOP_MSG,top);
    }
}
