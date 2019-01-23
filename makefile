# -*- mode: makefile-gmake -*-

override PETSCOPT_DIR := $(CURDIR)
include ./lib/petscopt/conf/variables
-include makefile.in
ruler := $(subst -,==========,--------)

all-parallel:
	-@echo $(ruler)
	-@echo "Building PETSCOPT (GNU Make - $(MAKE_NP) build jobs)"
	-@echo "Using PETSCOPT_DIR=$(PETSCOPT_DIR)"
	-@echo "Using PETSCOPT_ARCH=$(PETSCOPT_ARCH)"
	-@echo "Using PETSC_DIR=$(PETSC_DIR)"
	-@echo "Using PETSC_ARCH=$(PETSC_ARCH)"
	-@echo $(ruler)
	 @$(OMAKE) -j $(MAKE_NP) all
	-@echo $(ruler)
.PHONY: all-parallel

OBJDIR := $(PETSCOPT_ARCH)/obj
MODDIR := $(PETSCOPT_ARCH)/include
LIBDIR := $(abspath $(PETSCOPT_ARCH)/lib)

libpetscopt_shared := $(LIBDIR)/libpetscopt.$(SL_LINKER_SUFFIX)
libpetscopt_static := $(LIBDIR)/libpetscopt.$(AR_LIB_SUFFIX)
libpetscopt := $(if $(filter-out no,$(BUILDSHAREDLIB)),$(libpetscopt_shared),$(libpetscopt_static))
alllib : config $(generated) $(libpetscopt)
.PHONY: alllib

all : alllib

ifeq ($(V),)           # Print help and short compile line
  quiet_HELP := "Use \"$(MAKE) V=1\" to see the verbose compile lines.\n"
  quiet = @printf $(quiet_HELP)$(eval quiet_HELP:=)"  %-10s %s\n" "$1$2" "$@"; $($1)
else ifeq ($(V),0)     # Same as previous, but do not print any help
  quiet = @printf "  %-10s %s\n" "$1$2" "$@"; $($1)
else                   # Show the full command line
  quiet = $($1)
endif

pcc = $(if $(findstring CONLY,$(PETSC_LANGUAGE)),CC,CXX)
COMPILE.cc = $(call quiet,$(pcc)) $(PCC_FLAGS) $(CFLAGS) $(CCPPFLAGS) $(C_DEPFLAGS) -c
COMPILE.cpp = $(call quiet,CXX) $(CXX_FLAGS) $(CXXFLAGS) $(CCPPFLAGS) $(CXX_DEPFLAGS) -c
ifneq ($(FC_MODULE_OUTPUT_FLAG),)
COMPILE.fc = $(call quiet,FC) $(FC_FLAGS) $(FFLAGS) $(FCPPFLAGS) $(FC_DEPFLAGS) $(FC_MODULE_OUTPUT_FLAG)$(MODDIR) -c
else
FCMOD = cd $(MODDIR) && $(FC)
COMPILE.fc = $(call quiet,FCMOD) $(FC_FLAGS) $(FFLAGS) $(FCPPFLAGS) $(FC_DEPFLAGS) -c
endif

generated := $(PETSCOPT_DIR)/$(PETSCOPT_ARCH)/lib/petscopt/conf/files

.SECONDEXPANSION: # to expand $$(@D)/.DIR

write-variable = @printf "override $2 = $($2)\n" >> $1;
write-confheader-pre    = @printf "\#if !defined(__PETSCOPTCONF_H)\n\#define __PETSCOPTCONF_H\n" >> $1;
write-confheader-define = @printf "\n\#if !defined(PETSCOPT_HAVE_$2)\n\#define PETSCOPT_HAVE_$2 1\n\#endif\n" >> $1;
write-confheader-post   = @printf "\n\#endif\n" >> $1;
petscopt-conf-root = $(PETSCOPT_ARCH)
config-petsc      := $(petscopt-conf-root)/lib/petscopt/conf/config-petsc
config-petscopt   := $(petscopt-conf-root)/lib/petscopt/conf/config-petscopt
config-mfem       := $(petscopt-conf-root)/lib/petscopt/conf/config-mfem
config-confheader := $(petscopt-conf-root)/include/petscoptconf.h
$(config-petsc) : | $$(@D)/.DIR
	$(call write-variable,$@,PETSC_DIR)
	$(call write-variable,$@,PETSC_ARCH)
$(config-petscopt) : | $$(@D)/.DIR
	$(call write-variable,$@,CFLAGS)
	$(call write-variable,$@,CXXFLAGS)
$(config-mfem) : | $$(@D)/.DIR
	$(call write-variable,$@,with_mfem)
	$(call write-variable,$@,MFEM_DIR)
$(config-confheader) : | $$(@D)/.DIR
	$(call write-confheader-pre,$@)
ifeq ($(with_mfem),1)
	$(call write-confheader-define,$@,MFEMOPT)
endif
	$(call write-confheader-post,$@)

config-petsc : $(config-petsc)
config-petscopt : $(config-petscopt)
config-mfem : $(config-mfem)
config-confheader : config-vars $(config-confheader)

config-vars : config-petsc config-petscopt config-mfem
config : config-confheader
config-clean : clean
	$(RM) $(config-confheader) $(config-petsc) $(config-petscopt) $(config-mfem)
	$(RM) $(generated)
.PHONY:  config-petsc config-petscopt config-mfem config-vars config-confheader config config-clean

spkgs := ts,tao,mfemopt
pkgs := ts tao mfemopt
langs := c cpp

$(generated) : $(config-confheader) $(petscconf) $(petscvariables) $(PETSC_DIR)/config/gmakegen.py | $$(@D)/.DIR
	$(PYTHON) $(PETSC_DIR)/config/gmakegen.py --petsc-arch=$(PETSC_ARCH) --pkg-dir=$(PETSCOPT_DIR) --pkg-name=petscopt --pkg-arch=$(PETSCOPT_ARCH) --pkg-pkgs=$(spkgs)

-include $(generated)

concatlang = $(foreach lang, $(langs), $(srcs-$(1).$(lang):src/%.$(lang)=$(OBJDIR)/%.o))
srcs.o := $(foreach pkg, $(pkgs), $(call concatlang,$(pkg)))

objects.o := $(srcs.o)
.SECONDARY: $(objects.o)

$(libpetscopt_static) : objs := $(objects.o)
$(libpetscopt_shared) : objs := $(objects.o)
ifeq ($(with_mfem),1)
$(libpetscopt_shared) : libs := $(PETSC_LIB) $(MFEM_LIBS)
else
$(libpetscopt_shared) : libs := $(PETSC_LIB)
endif

%.$(SL_LINKER_SUFFIX) : $$(objs) | $$(@D)/.DIR
ifeq ($(with_mfem),1)
	$(call quiet,CXXLINKER) -shared -o $@ $^ $(libs)
else
	$(call quiet,CLINKER) -shared -o $@ $^ $(libs)
endif
ifneq ($(DSYMUTIL),true)
	$(call quiet,DSYMUTIL) $@
endif

%.$(AR_LIB_SUFFIX) : $$(objs) | $$(@D)/.DIR
ifeq ($(findstring win32fe lib,$(AR)),)
	@$(RM) $@
	$(call quiet,AR) $(AR_FLAGS) $@ $^
	$(call quiet,RANLIB) $@
else
	@$(RM) $@ $@.args
	@cygpath -w $^ > $@.args
	$(call quiet,AR) $(AR_FLAGS) $@ @$@.args
	@$(RM) $@.args
endif

$(OBJDIR)/%.o : src/%.c | $$(@D)/.DIR
	$(COMPILE.cc) $(abspath $<) -o $@

$(OBJDIR)/%.o : src/%.cpp | $$(@D)/.DIR
	$(COMPILE.cpp) $(abspath $<) -o $@

$(OBJDIR)/%.o : src/%.F90 | $$(@D)/.DIR $(MODDIR)/.DIR
	$(COMPILE.fc) $(abspath $<) -o $(if $(FCMOD),$(abspath $@),$@)

%/.DIR :
	@$(MKDIR) $(@D)
	@touch $@

.PRECIOUS: %/.DIR
.SUFFIXES: # Clear .SUFFIXES because we don't use implicit rules
.DELETE_ON_ERROR: # Delete likely-corrupt target file if rule fails
.PHONY: all check clean distclean install

check :
	-@echo $(ruler)
	-@echo "Running test examples to verify correct installation"
	-@echo "Using PETSCOPT_DIR=$(PETSCOPT_DIR)"
	-@echo "Using PETSCOPT_ARCH=$(PETSCOPT_ARCH)"
	-@echo "Using PETSC_DIR=$(PETSC_DIR)"
	-@echo "Using PETSC_ARCH=$(PETSC_ARCH)"
	-@echo $(ruler)
	+@cd ${PETSCOPT_DIR}/src/ts/examples/tests >/dev/null && $(RM) -f ex1 && ${OMAKE} PETSCOPT_ARCH=${PETSCOPT_ARCH}  PETSCOPT_DIR=${PETSCOPT_DIR} ex1
ifeq ($(with_mfem),1)
	+@cd ${PETSCOPT_DIR}/src/mfemopt/examples/tests >/dev/null && $(RM) -f ex1 && ${OMAKE} PETSCOPT_ARCH=${PETSCOPT_ARCH}  PETSCOPT_DIR=${PETSCOPT_DIR} ex1
endif
	-@echo $(ruler)

clean :
	$(RM) -r $(OBJDIR) $(LIBDIR)/libpetscopt*.*
distclean :
	@echo "*** Deleting all build files ***"
	-$(RM) -r $(PETSCOPT_DIR)/$(PETSCOPT_ARCH)/

#PREFIX = /tmp/petscopt
#find-install-dir = \
#find $2 -type d -exec \
#install -m $1 -d "$(PREFIX)/{}" \;
#find-install = \
#find $2 -type f -name $3 -exec \
#install -m $1 "{}" "$(PREFIX)/$(if $4,$4,{})" \;
#install :
#	@echo "*** Installing PETSCOPT in PREFIX=$(PREFIX) ***"
#	@$(call find-install-dir,755,bin)
#	@$(call find-install-dir,755,include)
#	@$(call find-install-dir,755,lib)
#	@$(call find-install,755,bin,'petscopt-*')
#	@$(call find-install,644,include,'*.h')
#	@$(call find-install,644,lib/petscopt/app,'*')
#	@$(call find-install,644,lib/petscopt/conf,'*')
#	@$(call find-install,644,lib/petscopt/python,'*.py')
#	@$(call find-install,644,$(PETSCOPT_ARCH)/lib,'libpetscopt*.$(AR_LIB_SUFFIX)',lib)
#	@$(call find-install,755,$(PETSCOPT_ARCH)/lib,'libpetscopt*.$(SL_LINKER_SUFFIX)',lib)
#	@printf "override PETSCOPT_ARCH =\n" > $(PREFIX)/lib/petscopt/conf/arch;
#	@$(OMAKE) petscopt-conf-root=$(PREFIX) config;
#	@$(RM) $(PREFIX)/lib/petscopt/conf/.DIR;

# TAGS generation
alletags :
	-@$(PYTHON) $(PETSC_DIR)/lib/petsc/bin/maint/generateetags.py
deleteetags :
	-@$(RM) CTAGS TAGS
allctags :
	-@ctags -o $(PETSC_DIR)/.vimctags -R --exclude=$(PETSC_DIR)/include/petsc/finclude $(PETSC_DIR)/src/ $(PETSC_DIR)/include/
	-@ctags -o .vimctags -R src/ include/
deletectags :
	-@$(RM) $(PETSC_DIR)/.vimctags
	-@$(RM) .vimctags
.PHONY: alletags deleteetags allctags deletectags

# make print VAR=the-variable
print : ; @echo "$($(VAR))"
.PHONY: print
# make print-VARIABLE
print-% : ; @echo "$* = $($*)"
.PHONY: print-%

help-make:
	-@echo
	-@echo "Basic build usage:"
	-@echo "   make -f ${makefile} <options>"
	-@echo
	-@echo "Options:"
	-@echo "  V=0           Very quiet builds"
	-@echo "  V=1           Verbose builds"
	-@echo

help-targets:
	-@echo "All makefile targets and their dependencies:"
	-@grep ^[a-z] makefile | grep : | grep -v =
	-@echo
	-@echo

help-test:
	-@echo "Basic test usage:"
	-@echo "   make test <options>"
	-@echo
	-@echo "Options:"
	-@echo "  NO_RM=1           Do not remove the executables after running"
	-@echo "  REPLACE=1         Replace the output in PETSC_DIR source tree (-m to test scripts)"
	-@echo "  ALT=1             Replace 'alt' output in PETSC_DIR source tree (-M to test scripts)"
	-@echo "  DIFF_NUMBERS=1    Diff the numbers in the output (-j to test scripts and petscdiff)"
	-@echo "  VALGRIND=1        Execute the tests using valgrind (-V to test scripts)"
	-@echo "  NP=<num proc>     Set a number of processors to pass to scripts."
	-@echo "  FORCE=1           Force SKIP or TODO tests to run"
	-@echo "  TIMEOUT=<time>    Test timeout limit in seconds (default in config/petsc_harness.sh)"
	-@echo "  TESTDIR='tests'   Subdirectory where tests are run ($${PETSC_DIR}/$${PETSC_ARCH}/$${TESTDIR}"
	-@echo "                    or $${PREFIX_DIR}/$${TESTDIR}"
	-@echo "                    or $${PREFIX_DIR}/share/petsc/examples/$${TESTDIR})"
	-@echo "  TESTBASE='tests'   Subdirectory where tests are run ($${PETSC_DIR}/$${PETSC_ARCH}/$${TESTDIR}"
	-@echo "  OPTIONS='<args>'  Override options to scripts (-a to test scripts)"
	-@echo "  EXTRA_OPTIONS='<args>'  Add options to scripts (-e to test scripts)"
	-@echo
	-@echo "Tests can be generated by searching:"
	-@echo "  Percent is a wildcard (only one allowed):"
	-@echo "    make -f ${makefile} test search=sys%ex2"
	-@echo
	-@echo "  To match internal substrings (matches *ex2*):"
	-@echo "    make -f ${makefile} test searchin=ex2"
	-@echo
	-@echo "  Search and searchin can be combined:"
	-@echo "    make -f ${makefile} test search='sys%' searchin=ex2"
	-@echo
	-@echo "  To match patterns in the arguments:"
	-@echo "    make -f ${makefile} test argsearch=cuda"
	-@echo
	-@echo "  For general glob-style searching using python:"

objects.d := $(objects.o:%.o=%.d)
# Tell make that objects.d are all up to date. Without
# this, the include below has quadratic complexity.
$(objects.d) : ;
-include $(objects.d)

# Handle test framework (also defines compilation lines for both lib and test)
include ./makefile.test        # This must be below the all target
