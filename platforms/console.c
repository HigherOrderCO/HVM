// IO platform supporting character-based console IO on UNIX-y systems

// This file is designed to be concatenated to the generated runtime

// Protocol
// ---------------------------------------------------------------------
// Program halts with term         Platform will
// -----------------------------   ----------------------------------------
//
//  Console.put_char c rest         1. Print HVM-int c to stdout (as a character)
//                                  2. Replace term with rest
//  (c a fully-evalauted int)       3. Tell runtime to continue
//
//  Console.get_char cont           1. Read a character c from stdin
//                                  2. Replace term with (cont c)
//                                  3. Tell runtime to continue
//
//  Console.done                    1. Read-back HVM-expression e to C-string
//                                  2. Print to stdout
//                                  3. Tell runtime to halt
//
//  <anything else>             Undefined behaviour.

// Implementation
// --------------

// Programs may not use all the special constructors
// In that case, these #define s make the platfrom still compile

#ifndef _CONSOLE_PUT__CHAR_
#define _CONSOLE_PUT__CHAR_ -1
#endif
#ifndef _CONSOLE_GET__CHAR_
#define _CONSOLE_GET__CHAR_ -2
#endif
#ifndef _CONSOLE_DONE_
#define _CONSOLE_DONE_ -3
#endif


#include <stdio.h>
#include <sys/time.h>

void io_setup() {
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

bool fail(char* msg, Ptr term){
    fprintf(stderr,"[Console IO platform] %s\n",msg);
    print_term(stderr,mem,term);
    return false;
}

char* BAD_TOP_MSG = "Illegal IO request!\nExpected one of the constructors {Console.put_char/2, Console.get_char/1, Console.done/1}, instead got:";

bool io_step() {
    Ptr top = mem->node[0];

    if(get_tag(top) != CTR) return fail(BAD_TOP_MSG,top);

    switch (get_ext(top)) {

        case _CONSOLE_DONE_:
            print_term(stdout,mem, ask_arg(mem,top,1));
            // TODO: decrement IO.done node?
        return false;

        case _CONSOLE_PUT__CHAR_:
            Ptr num_cell  = ask_arg(mem,top,0);
            Ptr rest_cell = ask_arg(mem,top,1);

            // Step 1: extract character & print
            if(!(get_tag(num_cell) == NUM && get_num(num_cell) <= 256))
                return fail("Bad number cell.\nExpected an int <= 256, instead got:",num_cell);
            printf("%c", (char)get_num(num_cell));

            // Step 2: replace top term with rest
            link(mem, 0, rest_cell);        // Overwrite [CTR|IO.do_output|...] (the root) with [rest_cell], and update backpointers
            // TODO: Free the constructor-data node (two consecutive cells: [num_cell][rest_cell])

        return true;

        case _CONSOLE_GET__CHAR_:

            // Step 1: Read a character c from stdin
            char c;
            fread((void*)&c,1,1,stdin);

            // Step 2: Replace term with (cont c)
            Ptr cont_cell = ask_arg(mem,top,0);

            Ptr app_data = alloc(mem,2);        // allocate cell for fun,arg pair
            link(mem,0,App(app_data));          // make main term an App-ptr
            link(mem,app_data+0,cont_cell);     // setup pair.fun
            link(mem,app_data+1,Num((u64)c));   // setup pair.arg

            // TODO: Free the constructor-data node (one cell: [cont_cell])

        return true;

        default:
        return fail(BAD_TOP_MSG,top);
    }
}
