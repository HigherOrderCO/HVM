// IO platform supporting character-based console IO on UNIX-y systems

// This file is designed to be concatenated to the generated runtime

// Protocol
// ---------------------------------------------------------------------
// (Main args...) reduces to       Platform will
// -----------------------------   ----------------------------------------
//
//  Console.put_char e rest         1. Evaluate e to whnf  (expect an int c)
//                                  2. Print HVM-int c to stdout (as a character)
//                                  2. Replace term with rest
//                                  3. Reduce again
//
//  Console.get_char cont           1. Read a character c from stdin
//                                  2. Replace term with (cont c)
//                                  3. Reduce again
//
//  Console.done                    1. Read-back HVM-expression e to C-string
//                                  2. Print to stdout
//                                  3. Reduce again
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
// (platform should not compile if _MAIN_ is missing)

#include <stdio.h>
#include <sys/time.h>
#include <stdbool.h>

Worker* mem;

// Debug helpers
static char* str_unknown = "???";

char * decode_cid(u64 cid){
    if (cid < id_to_name_size) {
        return id_to_name_data[cid];
    } else {
        return str_unknown;
    }
}

void print_term (FILE * stream, Ptr ptr) {
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
    print_term(stderr,term);
    return false;
}

char* BAD_TOP_MSG = "Illegal IO request!\nExpected one of the constructors {Console.put_char/2, Console.get_char/1, Console.done/1}, instead got:";

void debug_print_lnk(Ptr x) {
  u64 tag = get_tag(x);
  u64 ext = get_ext(x);
  u64 val = get_val(x);
  u64 num = get_num(x);
  switch (tag) {
    case DP0: fprintf(stderr,"[DP0 | 0x%-20"PRIx64" | 0x%-20"PRIx64" ]\n",  ext,    val ); break;
    case DP1: fprintf(stderr,"[DP1 | 0x%-20"PRIx64" | 0x%-20"PRIx64" ]\n",  ext,    val ); break;
    case VAR: fprintf(stderr,"[VAR | %22s | 0x%-20"PRIx64" ]\n",            "",     val ); break;
    case ARG: fprintf(stderr,"[ARG | %22s | 0x%-20"PRIx64" ]\n",            "",     val ); break;
    case ERA: fprintf(stderr,"[ERA | %22s | %22s ]\n",                      "",     ""  ); break;
    case LAM: fprintf(stderr,"[LAM | %22s | 0x%-20"PRIx64" ]\n",            "",     val ); break;
    case APP: fprintf(stderr,"[APP | %22s | 0x%-20"PRIx64" ]\n",            "",     val ); break;
    case SUP: fprintf(stderr,"[SUP | 0x%-20"PRIu64" | 0x%-20"PRIx64" ]\n",  ext,    val ); break;
    case CTR: fprintf(stderr,"[CTR | %22s | 0x%-20"PRIx64" ]\n", decode_cid(ext),   val ); break;
    case FUN: fprintf(stderr,"[FUN | %22s | 0x%-20"PRIx64" ]\n", decode_cid(ext),   val ); break;
    case OP2: fprintf(stderr,"[OP2 | 0x%-20"PRIu64" | 0x%-20"PRIx64" ]\n",  ext,    val ); break;
    case NUM: fprintf(stderr,"[NUM | %47"PRIu64" ]\n",                              num ); break;
    default : fprintf(stderr,"[??? | 0x%45"PRIx64" ]\n",                            num ); break;
  }
}

void dump(){
    for (u64 i = 0; i < mem->size; i ++){
        fprintf(stderr,"0x%-5"PRIx64"",i);
        debug_print_lnk(mem->node[i]);
    }
}

// Main

bool io_step() {
    Ptr top = mem->node[0];

    if(get_tag(top) != CTR) return fail(BAD_TOP_MSG,top);

    switch (get_ext(top)) {

        case _CONSOLE_DONE_:{
            print_term(stdout, ask_arg(mem,top,0));
            // TODO: decrement IO.done node?
        return false;}

        case _CONSOLE_PUT__CHAR_:{
            // Step 1: evaluate character
            u64 num_cell_p = get_loc(top,0);
            whnf(mem,num_cell_p);

            // Step 2: extract character & print
            Ptr num_cell  = ask_lnk(mem,num_cell_p);
            if(!(get_tag(num_cell) == NUM && get_num(num_cell) <= 256))
                return fail("Bad number cell.\nExpected an int <= 256, instead got:",num_cell);
            printf("%c", (char)get_num(num_cell));

            // Step 3: replace top term with rest
            Ptr rest_cell = ask_arg(mem,top,1);
            link(mem, 0, rest_cell);        // Overwrite [CTR|IO.do_output|...] (the root) with [rest_cell], and update backpointers
            // TODO: Free the constructor-data node (two consecutive cells: [num_cell][rest_cell])

        return true;}

        case _CONSOLE_GET__CHAR_:{

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

        return true;}

        default:{
        return fail(BAD_TOP_MSG,top);}
    }
}

int main (int argc, char* argv[]){
    mem = malloc(sizeof(Worker));
    build_main_term_with_args(mem,_MAIN_,argc,argv);

    do {
        whnf(mem,0);
    } while(io_step());

    free(mem->node);
    free(mem);
    return 0;
}
